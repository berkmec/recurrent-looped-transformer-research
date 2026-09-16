"""Fresh read-only closure-time byte validation, not scientific reproduction.

Public API verify(root)->JSON-serializable receipt. No files written, no model
imports/inference/training. PASS means all available required pins match; absent
historical whole-file pins remain explicitly absent in missing_prior_file_pins.
The caller owns exclusive receipt persistence and the campaign closure binding.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib,json,sqlite3,sys,time
from pathlib import Path

KNOWN_PROTOCOLS={
 'learning/protocol.json':'2647a34ff4a0c23fe2f275d1ac99e76906046a69f96c5f6f1eedcefb43e1b62f',
 'memory/protocol.json':'cce658466765424b59af28da85638ea2f67ba33c83a1d696ff4384ac36fd35d8',
 'memory/heading_protocol.json':'a39b9bd27cea71ee98f73c8c0ed6ba2077e1211a005468de8a294f51bde18260',
 'learning/distillation/protocol.json':'ba435010d7df25e5e43e1f125e4b5e16ed6254e60c3db599964b19d280a6c1c9',
 'integration/connected/protocol.json':'0b61590618f799ecece783cac7d43c08521f7d2ae901228f293a1df1c5a4e2ac',
 'integration/reasoning-companion/protocol.json':'54e37e67b107af932ceb2a22381547885944cc1139e83210a2bd6d30107c9de9',
 'integration/scale/protocol.json':'54ee3d844cd21435b98563c8ae5d077da9b498ba867c33e660ee992d72e9b41d'}

def verify(root):
    """Validate current bytes against existing pins; return PASS/FAIL receipt.

    Missing required receipts, mismatched bytes, incomplete core guard/audit and
    files changing while checked yield FAIL. No new pin is invented for artifacts
    without a previous whole-file digest. Core's existing check_entry is required.
    """
    root=Path(root).resolve();started=time.perf_counter();startutc=datetime.now(timezone.utc).isoformat();checked={};receipts={};missing=[];failures=[];conditions=[];bytes_read=0;payload_count=0;core_status=None;cache={}
    def fail(message):raise ValueError(message)
    def require(ok,message):
        if not ok:fail(message)
    def path(value):
        p=Path(value)
        if not p.is_absolute():p=root/p
        p=p.resolve();require(p.is_relative_to(root),'path escapes root: '+str(value));require(p.is_file(),'missing required file: '+str(p));return p
    def digest(value):
        nonlocal bytes_read
        p=path(value);before=p.stat();signature=(before.st_size,before.st_mtime_ns)
        if p in cache:
            require(cache[p][0]==signature,'file changed during closure check: '+str(p));return cache[p][1]
        h=hashlib.sha256()
        with p.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):h.update(block);bytes_read+=len(block)
        after=p.stat();require(signature==(after.st_size,after.st_mtime_ns),'file changed while hashing: '+str(p));cache[p]=(signature,h.hexdigest());return h.hexdigest()
    def pin(value,expected,authority):
        p=path(value);require(isinstance(expected,str) and len(expected)==64,'missing/invalid prior SHA: '+str(value));actual=digest(p);require(actual==expected,'PIN_MISMATCH '+str(p.relative_to(root))+' via '+authority);key=p.relative_to(root).as_posix();item=checked.setdefault(key,{'sha256':actual,'bytes':p.stat().st_size,'authorities':[]});item['authorities'].append(authority)
    def read(value):
        p=path(value);raw=p.read_bytes();sha=hashlib.sha256(raw).hexdigest();require(digest(p)==sha,'receipt changed while reading: '+str(p));receipts[p.relative_to(root).as_posix()]=sha;return json.loads(raw.decode('utf-8-sig'))
    def lines(value):
        p=path(value);digest(p)
        with p.open(encoding='utf-8-sig') as stream:
            for line in stream:
                if line.strip():yield json.loads(line)
        digest(p)
    def maps(base,mapping,authority):
        for name,expected in mapping.items():pin(base/Path(name),expected,authority)
    def unique(pattern):
        ps=sorted(root.glob(pattern));require(len(ps)==1,'expected one completed audit matching '+pattern);return ps[0]
    def audit(value,allowed=('PASS','PASS_WITH_SCOPE_LIMITS')):
        a=read(value);require(a.get('status') in allowed,'independent audit not passing: '+str(value));return a
    def selfhashed_db(relative):
        nonlocal payload_count
        p=path(relative);before=digest(p);db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
        try:
            db.execute('PRAGMA query_only=ON');tables={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ('records','compacts','observations'):
                if table not in tables:continue
                for rid,raw,expected in db.execute('SELECT id,raw,sha FROM '+table):
                    require(hashlib.sha256(raw.encode()).hexdigest()==expected,'stored payload PIN_MISMATCH '+relative+':'+table+':'+str(rid));payload_count+=1
        finally:db.close()
        require(digest(p)==before,'archive changed during payload checks');missing.append({'file':relative,'missing':'prior whole-file SHA256','available_check':'stored raw/compact/observation payload SHA256 verified; not equivalent to independent whole-archive pin'})
    try:
        # Completion audit is mandatory; do not emit PASS while core is pending.
        ca=audit('audit/core-independent-audit.json',('PASS',));require(ca.get('checked_files'),'core audit has no checked_files pins')
        maps(root/'core',ca['checked_files'],'independent core audit checked_files')
        if ca.get('audit_source_sha256'):pin('audit/verify_core_independent.py',ca['audit_source_sha256'],'core independent audit source receipt')
        binding=read('core/evidence-binding.json');maps(root/'core',binding['files'],'core post-run consistency binding (not pre-run signature)')
        guard=path('core/safe_entrypoint.py');require('safe_entrypoint.py' in binding['files'],'core guard not bound');pin(guard,binding['files']['safe_entrypoint.py'],'core guard binding before loading read-only check_entry')
        # Execute only the standard-library guard definitions; its main/binder
        # are never invoked. Disable bytecode creation during its numpy import.
        namespace={'__name__':'_closure_readonly_core_guard','__file__':str(guard)}
        old_bytecode=sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode=True;exec(compile(guard.read_text(),str(guard),'exec'),namespace);core_status=namespace['check_entry'](root/'core')
        finally:sys.dont_write_bytecode=old_bytecode
        require(core_status=='VERIFIED_COMPLETE','core check_entry != VERIFIED_COMPLETE');conditions.append('core_guard_and_independent_checked_files')
        for file,digest_value in KNOWN_PROTOCOLS.items():pin(file,digest_value,'known frozen protocol pin')
        for base in ['learning','learning/distillation','integration/connected','integration/reasoning-companion','memory']:
            side=root/base/'protocol.sha256'
            pin(root/base/'protocol.json',side.read_text().strip(),'recorded protocol SHA sidecar')
        # Primary learned-model receipts and independent prediction pins.
        learning=root/'learning';lp=read('learning/protocol.json');lex=list(lines('learning/execution.jsonl'));frozen=next(e for e in lex if e['event']=='protocol_frozen' and e.get('sha256')==KNOWN_PROTOCOLS['learning/protocol.json']);sources=read('learning/source-hashes.json');require(sources==frozen['sources'],'learning source index differs from pre-run receipt');maps(learning,sources,'learning pre-run frozen source receipt')
        lm=read('learning/results/metrics.json');cm=read('learning/results/combined-metrics.json');sm=read('learning/results/structural-extra-metrics.json');la=audit('audit/learning-independent-audit.json')
        require(la['raw_files_recomputed']==111 and len(lm['rows'])+len(cm['rows'])+len(sm['rows'])==111,'learning prediction completeness')
        for entry in la['per_file_metrics']:pin(learning/'results'/entry['file'],entry['sha256'],'independent learning raw-file audit')
        for source,protocol in [('combined_run.py','combined-protocol.json'),('structural_extra.py','structural-extra-protocol.json')]:
            pp=read(learning/protocol);pin(learning/source,pp['source_sha256'],'learning supplemental pre-run source pin');pin(learning/protocol,(learning/protocol.replace('.json','.sha256')).read_text().strip(),'supplemental protocol sidecar');require(pp['primary_protocol_sha256']==KNOWN_PROTOCOLS['learning/protocol.json'],'supplemental primary pin')
        for package in [lm,cm,sm]:
            for row in package['rows']:pin(row['raw_path'],row['raw_sha256'],'learning completed raw receipt')
        for package in [lm,cm]:
            for run in package['runs']:
                arm=run.get('arm','combined');seed=run['seed'];starts=[e for e in lex if e['event'] in ['training_start','combined_training_start'] and e.get('seed')==seed and (e.get('arm')==arm or (arm=='combined' and e['event']=='combined_training_start'))];pin(learning/'checkpoints'/f'{arm}-{seed}-initial.pt',starts[-1]['initial_sha256'],'learning initial checkpoint run receipt')
                for point in run['curve']:pin(learning/'checkpoints'/f"{arm}-{seed}-step{point['step']}.pt",point['checkpoint_sha256'],'learning checkpoint training receipt')
        for event in lex:
            if event['event'] in ['dataset_created','test_first_open'] and event.get('sha256'):
                # Earlier retained diagnostic TRAIN/validation arrays are shared
                # but diagnostic heldout seeds were never opened.
                if event.get('split') in ['train','validation','iid','shift']:pin(learning/'results'/f"data-{event['split']}.npz",event['sha256'],'learning dataset creation receipt')
            if event['event']=='structural_extra_test_first_open':pin('learning/results/data-depth8.npz',event['sha256'],'depth8 first-open receipt')
            if event['event']=='all_selection_frozen_before_test':pin('learning/revision-selection.json',event['selection_sha256'],'learning selection frozen before test');pin('learning/training.json',event['training_sha256'],'learning training frozen before test')
        conditions.append('learning_protocol_sources_checkpoints_111_raw_predictions')
        # Current memory authority: later pre-run connected/window pins below;
        # final-source manifest is explicitly a post-run receipt, not pre-run.
        mem=root/'memory';mf=read('memory/final-source-manifest.json');maps(root,mf['source_sha256'],'memory final-source post-run manifest');maps(mem/'checkpoints',mf['checkpoint_sha256'],'memory checkpoint final manifest');pin(mem/'protocol.json',mf['protocol_sha256'],'memory final manifest protocol')
        em=read('memory/results/extractor-metrics.json')
        for row in em:
            pin(mem/'checkpoints'/f"extractor-{row['seed']}.pt",row['checkpoint_sha256'],'extractor seed completion receipt');pin(mem/'checkpoints'/f"extractor-{row['seed']}-initial.pt",row['initial_sha256'],'extractor initial checkpoint receipt')
        hm=read('memory/heading-results/metrics.json');pin(mem/'heading_protocol.json',hm['protocol_sha256'],'heading measurement frozen protocol receipt');maps(root,read('memory/heading-results/manifest-final.json')['source_sha256'],'heading final-source post-run manifest')
        for relative in ['memory/archive.sqlite','memory/archive-v2.sqlite','memory/archive-v3.sqlite','memory/provenance-smoke.sqlite']:selfhashed_db(relative)
        for pattern in ['memory/results/extractor-*.jsonl','memory/results/extractor-*.npz','memory/results/memory-*.jsonl','memory/results/*predictions.npz','memory/results-v2/*.jsonl','memory/results-v2/*.npz','memory/results-v3/*.jsonl','memory/results-v3/*.npz','memory/heading-results/predictions.jsonl','memory/heading-results/heading-history.json']:
            for file in root.glob(pattern):missing.append({'file':file.relative_to(root).as_posix(),'missing':'recorded prior whole-file SHA256','available_check':'none in original scientific receipts; not counted as whole-file verified'})
        conditions.append('memory_available_source_checkpoint_protocol_and_payload_pins')
        # Distillation and separately frozen descriptive initial reference.
        d=root/'learning/distillation';dp=read(d/'protocol.json');maps(d,dp['sources'],'distillation pre-run source');maps(root,dp['external_sources'],'distillation pre-run external source');maps(learning/'checkpoints',dp['checkpoints'],'distillation pre-run teacher/initial checkpoint');pin(learning/'results/data-train.npz',dp['train_source_sha256'],'distillation TRAIN source pin');dep=read(d/'dependency-receipt.json');pin(d/'dependency-receipt.json',(d/'dependency-receipt.sha256').read_text().strip(),'distillation dependency sidecar');maps(root,dep['files'],'distillation transitive dependency receipt')
        da=audit('audit/distillation-independent-audit.json');require(not da['failures'],'distillation audit has failures');dm=read(d/'results/metrics.json')
        for seed in dp['seeds']:
            receipt=read(d/'archives'/f'receipt-{seed}.json')
            for filename,key in [(f'teacher-{seed}.npz','archive_sha256'),(f'selection-{seed}.npz','selection_sha256'),(f'decisions-{seed}.jsonl','decisions_sha256')]:pin(d/'archives'/filename,receipt[key],'distillation closed teacher archive receipt')
        for row in dm['rows']:pin(row['raw_path'],row['raw_sha256'],'distillation heldout raw receipt')
        for row in dm['runs']:pin(d/'checkpoints'/f"{row['arm']}-{row['seed']}-step300.pt",row['checkpoint_sha256'],'student completed checkpoint receipt')
        for e in lines(d/'execution.jsonl'):
            if e['event']=='all_student_selection_fixed_before_test':pin(d/'results/training.json',e['training_sha256'],'all student training frozen before test')
        off=read(d/'results/archive-off-receipt.json');pin(d/'inference-only-export/predictions.npz',off['predictions_sha256'],'student archive-off prediction receipt')
        for filename in off['models']:
            expected=next(r['checkpoint_sha256'] for r in dm['runs'] if filename==f"{r['arm']}-{r['seed']}-step300.pt");pin(d/'inference-only-export'/filename,expected,'archive-off actual student checkpoint')
        for filename in ['api.py','model.py','token_core.py','__init__.py']:pin(d/'inference-only-export/learning'/filename,sources[filename],'archive-off pinned inference source')
        ref=root/'audit/distillation-initial-reference';rp=read(ref/'protocol.json');pin(ref/'protocol.json',(ref/'protocol.sha256').read_text().strip(),'post-hoc reference separately frozen protocol');maps(root,rp['checkpoints'],'post-hoc initial checkpoint pre-inference pins');pin('learning/api.py',rp['model_api_sha256'],'post-hoc API pin');pin('audit/verify_distillation.py',rp['audit_source_sha256'],'post-hoc evaluator pre-inference pin')
        for split,expected in rp['datasets'].items():pin(d/'results'/f'data-{split}.npz',expected,'post-hoc source dataset pre-inference pin')
        for row in read(ref/'metrics.json')['rows']:pin(ref/f"pred-initial-{row['seed']}-{row['split']}.npz",row['prediction_sha256'],'post-hoc initial prediction completion receipt')
        conditions.append('distillation_archives_students_raw_and_initial_reference')
        # Connected/window audit -> prediction JSONL -> every archive hash chain.
        for folder,auditpattern,count,groupcount,kind in [('connected','audit/connected-independent-20??????T??????.json',6144,24,'connected'),('reasoning-companion','audit/reasoning-companion-independent-20??????T??????.json',1536,12,'window')]:
            adpath=unique(auditpattern);ad=audit(adpath);cp=read(root/'integration'/folder/'protocol.json');maps(root,cp['source_sha256'],folder+' pre-run current scientific source');maps(root,cp.get('upstream_protocol_sha256',{}),folder+' upstream protocol pins');maps(root,cp.get('checkpoint_sha256',{}),folder+' pre-run checkpoint pins');directory=root/'integration'/folder/'results';complete=read(directory/'metrics.json');require(len(complete['rows'])==groupcount and ad['executions']==count,'connected/window completeness')
            if kind=='connected':maps(root,read(directory/'environment.json')['checkpoint_sha256'],'connected launch checkpoint receipt')
            episodes=0
            for row in complete['rows']:
                name=row['condition']['name'] if kind=='connected' else row['condition'];label=f"{name}-{row['split']}-s{row['seed']}" if kind=='connected' else f"{row['split']}-{name}-s{row['seed']}";group=directory/label;pred=group/'predictions.jsonl';pin(pred,row['predictions_sha256'] if kind=='connected' else row['prediction_sha256'],folder+' completed prediction hash')
                if kind=='connected':prior=next(g for g in ad['groups'] if g['seed']==row['seed'] and g['split']==row['split'] and g['condition']==name);pin(pred,prior['predictions_sha256'],'independent connected audit prediction hash')
                for index,episode in enumerate(lines(pred)):pin(group/f'episode-{index:04d}.sqlite',episode['archive_sha256'],folder+' pinned prediction -> whole episode DB');episodes+=1
            require(episodes==count,'episode archive cardinality '+folder)
            for e in lines(directory/'execution.jsonl'):
                if e['event'] in ['fresh_test_first_open','fresh_test_opened']:pin(directory/f"data-{e['split']}.json",e['sha256'],folder+' first test opening source receipt')
            if ad.get('auditor_source_sha256'):pin('audit/verify_connected.py',ad['auditor_source_sha256'],'independent connected auditor source')
            if ad.get('source_sha256'):pin('audit/verify_reasoning_companion.py',ad['source_sha256'],'independent window auditor source')
            conditions.append(folder+'_all_prediction_and_archive_byte_pins')
        # Width256 scaling current file receipts and independent audit SHA rows.
        scale=root/'integration/scale';sp=read(scale/'protocol.json');smetrics=read(scale/'metrics.json');sa=audit('audit/scale-independent-audit.json',('PASS',));pin('core/model.py',sp['code_hashes']['model'],'scale pre-run model source');pin('integration/run_scale.py',sp['code_hashes']['runner'],'scale pre-run runner source');pin(scale/'protocol.json',smetrics['protocol_sha256'],'scale completed protocol receipt');require(len(smetrics['training'])==len(smetrics['evaluation'])==6 and sa['prediction_files_checked']==60,'scale completeness')
        for row in smetrics['training']:pin(scale/row['name']/'initial.pt',row['initial_file_sha256'],'scale initial file receipt');pin(row['selected_checkpoint'],row['selected_checkpoint_sha256'],'scale selected checkpoint receipt')
        for ev in smetrics['evaluation']:
            for family in ['long_horizon','neutral_slots']:
                for r in ev[family]:
                    pin(r['file'],r['sha256'],'scale final raw array receipt');index=r['length'] if family=='long_horizon' else r['slots'];ind=next(x for x in sa['rows'] if x['seed']==ev['seed'] and x['mode']==ev['mode'] and x['family']==family and x['index']==index);pin(r['file'],ind['sha256'],'independent scale audited raw SHA')
        conditions.append('scale_sources_initial_selected_checkpoints_60_raw_arrays')
        # Re-stat every hashed/read artifact at return; race-free at checked scope.
        for p,(signature,h) in cache.items():require((p.stat().st_size,p.stat().st_mtime_ns)==signature,'file changed before closure verifier return: '+str(p))
    except Exception as error:failures.append({'type':type(error).__name__,'message':str(error)})
    return {'status':'FAIL' if failures else 'PASS','scope':'Fresh byte consistency of available recorded pins only; not new scientific reproduction or complete historic whole-file pin coverage','utc_started':startutc,'utc_finished':datetime.now(timezone.utc).isoformat(),'core_check_entry':core_status,'verified_conditions':conditions,'failures':failures,'checked_file_count':len(checked),'checked_files':checked,'read_receipt_current_sha256':receipts,'missing_prior_file_pins':missing,'stored_memory_payload_pins_checked':payload_count,'whole_file_coverage_complete':not missing,'bytes_hashed':bytes_read,'seconds':time.perf_counter()-started,'limitations':['PASS does not convert missing historical whole-file pins into verified files. Missing prior pins are listed explicitly.','Memory final/heading manifests and core evidence-binding are post-run internal consistency receipts, not pre-run signed attestations. Current shared memory runtime sources additionally match later connected/window pre-run pins.','Existing independent audit receipts authenticate only the hashes they actually recorded; whole-file digests freshly computed while reading other receipts are receipt snapshots, not invented prior authority.','Core existing check_entry performs its required read-only completeness/array checks; no model inference/training or producer mutation occurs.','No filesystem writes by verify; caller exclusively persists this receipt immediately before its own final closure.','End-of-check size/mtime revalidation detects ordinary concurrent mutation within this invocation; caller remains responsible for closure-time quiescence.']}
