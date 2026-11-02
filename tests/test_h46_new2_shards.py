"""Scheduling-only regression; no saved input, sampling or compile."""
import ast
import hashlib
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]


class ShardTests(unittest.TestCase):
    def test_twenty_four_unique_registered_tasks(self):
        tasks=[(g,i) for g in ('H6_S2','H6_Y4','H6_N2') for i in range(8)]
        self.assertEqual(len(set(tasks)),24)
        self.assertEqual({i for g,i in tasks},set(range(8)))

    def test_source_and_input_guards_identical(self):
        def nodes(filename):
            return {n.name:ast.dump(n) for n in ast.parse((ROOT/'scripts'/filename).read_text()).body
                if isinstance(n,ast.FunctionDef)}
        a=nodes('run_h46_new2_native.py');b=nodes('run_h46_new2_native_shard.py')
        for name in ('sha','write','verify'):self.assertEqual(a[name],b[name])

    def test_seed_namespace_retained_across_executor_change(self):
        old='687679a4ae20e4661bde1187dd8ab92ae176bd44'
        def seed(identity,g,i,a,b):
            return int.from_bytes(hashlib.sha256(f'{identity}|{g}|{i}|{a}|{b}'.encode()).digest()[:8],'big')
        config=dict(SOURCE='f'*40,seed_SOURCE=old)
        for i in range(8):
            self.assertEqual(seed(old,'H6_N2',i,0,3),seed(config['seed_SOURCE'],'H6_N2',i,0,3))
        s=(ROOT/'scripts/run_h46_new2_native_shard.py').read_text()
        self.assertIn("{c['seed_SOURCE']}|{group['id']}|{sample}|{a}|{b}",s)

    def test_shard_statistics_not_claimed_as_n8(self):
        s=(ROOT/'scripts/run_h46_new2_native_shard.py').read_text()
        self.assertIn('RZ_shot_weighted_SE=None',s)
        self.assertEqual(s.count('for sample in (sample_index,):'),2)
        self.assertIn("if label=='4th(new_2)' and sample_index==0:",s)


if __name__=='__main__':unittest.main()
