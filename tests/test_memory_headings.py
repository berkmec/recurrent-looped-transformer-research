import unittest,json
import tempfile
from pathlib import Path
from memory.headings import HeadingRegistry

def event(kind='root',child='x0',parent=None,value=1): return dict(kind=kind,child=child,parent=parent,value=value)

class HeadingTests(unittest.TestCase):
    def test_changed_wording_does_not_duplicate_title(self):
        r=HeadingRegistry(); a=r.consume('A',{**event(),'source_text':'set x0 to one'}); b=r.consume('A',{**event(),'source_text':'x0 starts at one'})
        self.assertEqual(a['heading_id'],b['heading_id']); self.assertEqual(b['version'],1); self.assertEqual(b['status'],'UNCHANGED'); self.assertEqual(len(r.headings('A')),1)
    def test_revision_keeps_identity_and_history(self):
        r=HeadingRegistry(); first=r.consume('A',event()); second=r.consume('A',event('revision',value=2)); h=r.get('A','x0')
        self.assertEqual(first['heading_id'],second['heading_id']); self.assertEqual(h['version'],2); self.assertEqual(h['versions'][0]['facets']['absolute']['value'],1); self.assertEqual(h['versions'][1]['facets']['absolute']['value'],2)
        h['versions'][0]['facets']['absolute']['value']=99; self.assertEqual(r.get('A','x0')['versions'][0]['facets']['absolute']['value'],1)
    def test_new_entity_creates_new_heading_and_typed_link(self):
        r=HeadingRegistry(); a=r.consume('A',event()); b=r.consume('A',event('edge','x1','x0',.5)); self.assertNotEqual(a['heading_id'],b['heading_id']); self.assertEqual(len(r.headings('A')),2)
        link=r.active_links('A')[0]; self.assertEqual(link['type'],'DEPENDS_ON_AFFINE'); self.assertEqual(link['source'],b['heading_id']); self.assertEqual(link['target'],a['heading_id']); self.assertEqual(link['offset'],.5)
    def test_task_isolation_query_and_missing(self):
        r=HeadingRegistry(); a=r.consume('A',event()); b=r.consume('B',event()); self.assertNotEqual(a['heading_id'],b['heading_id'])
        q=r.consume('A',event('query',value=None)); self.assertEqual(q['status'],'QUERY_RESOLVED'); self.assertEqual(q['heading_id'],a['heading_id'])
        missing=r.consume('C',event('query',value=None)); self.assertEqual(missing['status'],'MISSING_ENTITY'); self.assertEqual(r.headings('C'),[])
    def test_reference_heading_and_independent_facets(self):
        r=HeadingRegistry(); r.consume('A',event('edge','x1','x0',.5)); self.assertEqual(r.get('A','x0')['version'],0)
        r.consume('A',event('root','x1',None,2)); h=r.get('A','x1'); self.assertEqual(set(h['versions'][-1]['facets']),{'absolute','edge'}); self.assertEqual(len(r.active_links('A')),1)
    def test_unsupported_no_mutation_and_roundtrip(self):
        r=HeadingRegistry(); r.consume('A',event()); before=r.headings('A'); result=r.consume('A',{'kind':'topic','child':'x2'}); self.assertEqual(result['status'],'UNSUPPORTED'); self.assertEqual(before,r.headings('A'))
        self.assertEqual(r.consume('A',event(child=''))['status'],'MISSING_ENTITY')
        restored=HeadingRegistry.from_serialized(json.loads(json.dumps(r.serialize()))); self.assertEqual(restored.serialize(),r.serialize())
    def test_parse_rejection_cannot_create_heading(self):
        r=HeadingRegistry(); result=r.consume_parsed('A',{'valid':False,'events':[event()],'abstained':True}); self.assertEqual(result['status'],'EXTRACTION_UNRESOLVED'); self.assertEqual(r.headings('A'),[])
    def test_controller_compactions_preserve_heading_identity(self):
        from memory.api import Archive,TitleController
        with tempfile.TemporaryDirectory() as folder:
            archive=Archive(Path(folder)/'a.sqlite'); controller=TitleController(3,archive); registry=HeadingRegistry(); ids=[]
            for i in range(13):
                e=event('root' if i==0 else 'revision',value=i); result=registry.consume('A',e); ids.append(result['heading_id']); controller.append('A',result['heading_id'],e)
            self.assertEqual(len(set(ids)),1); self.assertEqual(controller.compactions,4); self.assertEqual(registry.get('A','x0')['version'],13); self.assertEqual(len(controller.reconstruct('A',ids[0])),13)
            restored=HeadingRegistry.from_serialized(registry.serialize()); self.assertEqual(restored.get('A','x0')['id'],ids[0]); archive.close()

if __name__=='__main__': unittest.main()
