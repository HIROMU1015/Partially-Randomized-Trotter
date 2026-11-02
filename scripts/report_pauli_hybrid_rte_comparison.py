#!/usr/bin/env python3
"""Format completed saved results; no science kernels, sampling or ranking."""
import csv
import json
from pathlib import Path
from fractions import Fraction as Q

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/pauli_hybrid_rte_comparison/2026-10-25'


def number(x):return '—' if x is None else format(float(Q(str(x))), '.6g')
def parameters(x):return '—' if x is None else '/'.join(x[k] for k in ('L_D','q','r','K'))


def main():
    assert json.loads((OUT/'completion_verification.json').read_text())['status']=='PASS'
    data=json.loads((OUT/'comparison_by_precision.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    labels={'HYBRID_LOWER_EMPIRICAL_G':'Hybrid低G', 'RTE_LOWER_EMPIRICAL_G':'RTE低G',
            'EMPIRICAL_NEAR_TIE_OR_UNRESOLVED':'近接/未解決', 'CONFIRMATION_MISSING':'確認不足'}
    text=['## 確認結果と探索結果', '',
          '以下は仮想ρ=0、有限価格標本、保存会計可能な候補内の結果である。',
          'R=G_Hybrid/G_RTE。R<1はHybrid、R>1はRTEの点費用が低い。表示値は丸め、exact値はCSV/JSONに保持する。',
          '「低G」は名目2 combined-SE基準で近接とされなかった経験的分類であり、母平均winnerや選抜後の厳密な信頼保証ではない。', '',
          '| 分子 | ε | Hybrid L/q/r/K | RTE L/q/r/K | 確認n H/R | G Hybrid | G RTE | G SE H/R | R | 観測分類 |',
          '|---|---|---|---|---|---|---|---|---|---|']
    scope=[]
    for x in data:
        h=x['confirmation_Hybrid']['selected'];r=x['confirmation_RTE']['selected']
        unknown=x['exploration_Hybrid']['unresolved_ids']+x['exploration_RTE']['unresolved_ids']
        text.append('| '+' | '.join([x['molecule'],x['epsilon_sig'],parameters(h),parameters(r),
            (h['price_n'] if h else '—')+'/'+(r['price_n'] if r else '—'),
            number(h['G_exact'] if h else None),number(r['G_exact'] if r else None),
            number(h['G_SE'] if h else None)+'/'+number(r['G_SE'] if r else None),
            number(x['confirmation_Hybrid_over_RTE_exact']),labels[x['confirmation_classification']]+(' †' if unknown else '')])+' |')
        scope.append(dict(molecule=x['molecule'],epsilon_sig=x['epsilon_sig'],
            observed_finite_subset_classification=x['confirmation_classification'],
            eligible_exploration_G_complete=not unknown,
            unresolved_exploration_RTE_ids=';'.join(x['exploration_RTE']['unresolved_ids']),
            unresolved_exploration_Hybrid_ids=';'.join(x['exploration_Hybrid']['unresolved_ids']),
            complete_registered_family_winner_established=False,population_winner_established=False,
            scope='Finite saved-G subset and confirmed subset; H4 historical baseline separate'))
    text+=['','†：適格性/会計domain等が未解決の登録候補が残る。表示された有限G部分集合の点順位を、登録全体の確定順位とはしない。',
           '[精度別の未解決ID・比較scope](../artifacts/pauli_hybrid_rte_comparison/2026-10-25/comparison_scope_status.csv)を参照。', '',
           '| 分子 | ε | 探索Hybrid L/q/r/K | 探索RTE L/q/r/K | 探索G Hybrid | 探索G RTE | 探索R |',
           '|---|---|---|---|---|---|---|']
    for x in data:
        h=x['exploration_Hybrid']['selected'];r=x['exploration_RTE']['selected']
        text.append('| '+' | '.join([x['molecule'],x['epsilon_sig'],parameters(h),parameters(r),
            number(h['G_exact'] if h else None),number(r['G_exact'] if r else None),
            number(x['exploration_Hybrid_over_RTE_exact'])])+' |')
    text+=['', '探索と確認をpoolしていない。探索標本の小ささ、確認後選抜、候補ごとのn差を保持する。', '',
           '## 代表条件の費用内訳', '',
           'Cは軸ごとの平均compiled RZ。biasはsigned cosine/sine。BとNは保存値を維持し、Nやnormalizationを再計算していない。',
           '数値allowanceのexact値はcost_breakdown.csvのu_cos_exact/u_sin_exact列に保持する。', '',
           '| 分子 | ε | 群 | B | bias cos/sin | N cos/sin | C cos/sin |',
           '|---|---|---|---|---|---|---|']
    for x in data:
        if x['epsilon_sig'] not in ('0.05','0.0005'):continue
        for g in ('Hybrid','RTE'):
            r=x['confirmation_'+g]['selected']
            if r is None:continue
            text.append('| '+' | '.join([x['molecule'],x['epsilon_sig'],g,
                number(r['B_exact'] or r['B_saved_decimal']),number(r['bias_cos_exact'])+'/'+number(r['bias_sin_exact']),
                r['N_cos']+'/'+r['N_sin'],number(r['C_cos_exact'])+'/'+number(r['C_sin_exact'])])+' |')
    text+=['', '## 実行・照合', '',
           f"追加trajectoryは{verification['new_trajectory_calls']}本、両軸compileは{verification['new_axis_compiles']}回。新状態作用・Hamiltonian/state/reference生成は0。",
           '人工10件PASS。保存N/B/bias/allowance、seed、compiled QPYのSHA/RZ/CX/wire、native exit0/reapを照合した。',
           '新rawの可逆archiveと元restorerの復元SHA結果はdata_archives.json / archive_restore_verification.jsonを参照。', '']
    p=ROOT/'docs/pauli_hybrid_rte_comparison_20261025.md'
    content=p.read_text();marker='<!-- RESULTS_AFTER_COMPLETION -->'
    assert content.count(marker)==1
    p.write_text(content.replace(marker,'\n'.join(text)))
    with (OUT/'comparison_scope_status.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(scope[0]),lineterminator='\n');w.writeheader();w.writerows(scope)


if __name__=='__main__':main()
