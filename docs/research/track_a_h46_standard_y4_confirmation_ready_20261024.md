# H4/H6 標準Yoshida4 独立確認の準備

[固定契約・S2再利用・n設計・予算・技術検査・STOP](../../artifacts/resource_applicability/track_a_h46_standard_y4_confirmation_ready/2026-10-24/REPORT.md)。新Y4はH4 n45/H6 n32、旧独立S2はn25/23を再会計する。ε=.001、標準S2/q8対Y4/q4、rank10/19、PR prefix3/5、K4・outer32を固定した。新科学call・有効grantは0であり、一回のPhase C承認待ち。

sampling/native/compile/paired statisticsはc028の元実装を再利用する。新adapterは[scripts](../../scripts/h46_standard_y4_contract.py)、[runtime](../../scripts/h46_standard_y4_runtime.py)、[統計入力結合](../../scripts/h46_standard_y4_evaluation.py)、[prepare](../../scripts/prepare_h46_standard_y4.py)、[runner](../../scripts/run_h46_standard_y4.py)、[人工tests](../../tests/test_h46_standard_y4.py)。新規Y4 77軌道/154 compile軸、保存S2再取得0、会計9 event。判定は差の点方向と2比較familyの正規近似区間を区別し、INCONCLUSIVEを許容する。

nは保存pilot設計表の相対MCSE1%・分散倍率4を採用対象とした値であり、母分散上界・power・希少tail保証ではない。旧独立S2のraw/実使用registryの新回収を主張せず、受入済みpair CSVと成功監査を条件付きで再利用する。旧strict COMPARE BLOCKED、旧STOP保持。
