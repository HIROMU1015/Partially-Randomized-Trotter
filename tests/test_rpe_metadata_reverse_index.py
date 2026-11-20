"""Real synthetic Git packs verify reverse-index-aware lossless compaction."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from trottertracks import rpe_worker_storage as S


class ReverseIndexCompaction(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        self.repo=self.root/'objects.git'
        subprocess.run(['git','init','-q','--bare',str(self.repo)],check=True)
        self.pack=self.repo/'objects/pack'
        self.payload=b'synthetic metadata only'*8192
        self.object=self.git('hash-object','-w','--stdin',input=self.payload).stdout.strip()
        for compression in (0,9):
            self.git('-c','pack.writeReverseIndex=true','pack-objects','--compression='+str(compression),
                     '--no-reuse-object',str(self.pack/'pack'),input=self.object+b'\n')
        self.assertEqual(len(list(self.pack.glob('*.pack'))),2)
        self.assertEqual(len(list(self.pack.glob('*.rev'))),2)

    def git(self,*args,**kwargs):
        return subprocess.run(['git','-C',str(self.repo),*args],check=True,capture_output=True,**kwargs)

    def test_valid_paired_reverse_indexes_compact_without_object_loss(self):
        original=S._packed_objects(self.pack)
        before=sum(p.stat().st_blocks*512 for p in self.pack.iterdir())
        # Prove the resulting object is read from packs, not the original loose copy.
        (self.repo/'objects'/self.object[:2].decode()/self.object[2:].decode()).unlink()
        reclaimed=S._compact_pack_directory(self.pack,self.root)
        self.assertGreater(reclaimed,0)
        self.assertEqual(S._packed_objects(self.pack),original)
        self.assertEqual(self.git('cat-file','blob',self.object.decode()).stdout,self.payload)
        self.assertLess(sum(p.stat().st_blocks*512 for p in self.pack.iterdir()),before)
        for pack in self.pack.glob('*.pack'):
            self.git('index-pack','--verify','--rev-index',str(pack))

    def test_unpaired_sidecar_and_unknown_entries_preserve_original_pack_bytes(self):
        originals={p.name:p.read_bytes()for p in self.pack.iterdir()}
        extra=self.pack/'orphan.rev';extra.write_bytes(b'not a paired index')
        self.assertEqual(S._compact_pack_directory(self.pack,self.root),0)
        extra.unlink()
        extra=self.pack/'tmp_pack_unfinished';extra.write_bytes(b'owned unfinished bytes')
        self.assertEqual(S._compact_pack_directory(self.pack,self.root),0)
        self.assertEqual(extra.read_bytes(),b'owned unfinished bytes')
        self.assertTrue(all((self.pack/name).read_bytes()==raw for name,raw in originals.items()))

    def test_corrupt_paired_reverse_index_fails_native_verification_without_replacement(self):
        rev=next(self.pack.glob('*.rev'));rev.chmod(0o644);rev.write_bytes(b'corrupt')
        originals={p.name:p.read_bytes()for p in self.pack.iterdir()}
        with self.assertRaises(subprocess.CalledProcessError):
            S._compact_pack_directory(self.pack,self.root)
        self.assertTrue(all((self.pack/name).read_bytes()==raw for name,raw in originals.items()))


if __name__=='__main__':unittest.main()
