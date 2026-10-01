"""Shell scripts must use LF endings, because they are written on Windows and run on Linux.

Found by launching the pipeline: `set -euo pipefail` became `set -euo pipefail\r`, bash reported
"set: pipefail: invalid option name", and the whole run died one line in. The scripts were authored on this
machine, where CRLF is the default, and copied verbatim to a Debian box.

Stripping the carriage returns on the VM fixes that copy and nothing else - the next sync would bring them back.
So the check lives here, over the files as they are stored, and it also covers the python scripts: a `#!` line
ending in CR makes a file unexecutable in exactly the same way, and the failure reads as "no such file".
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LineEndingTests(unittest.TestCase):
    def offenders(self, pattern):
        return [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'scripts').glob(pattern))
                if b'\r\n' in p.read_bytes()]

    def test_shell_scripts_have_no_carriage_returns(self):
        self.assertEqual(self.offenders('*.sh'), [],
                         'CRLF in a shell script makes bash read `pipefail\r` as an option name')

    def test_scripts_meant_for_the_vm_have_no_carriage_returns(self):
        self.assertEqual(self.offenders('*.py'), [],
                         'a shebang line ending in CR makes the file unexecutable, reported as "no such file"')

    def test_the_shell_scripts_still_start_with_a_shebang(self):
        for path in sorted((ROOT / 'scripts').glob('*.sh')):
            first = path.read_bytes().split(b'\n', 1)[0]
            self.assertTrue(first.startswith(b'#!'), f'{path.name} lost its shebang')
            self.assertNotIn(b'\r', first, f'{path.name} shebang carries a carriage return')


if __name__ == '__main__':
    unittest.main()
