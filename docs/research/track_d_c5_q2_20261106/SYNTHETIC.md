# C5人工・回帰検証

C5新規8件、既存C3 4件、C2 2件、Stage A 2件の計16件がPASSした。2-mode人工系・mock input・小さいoperatorだけを使用し、分子prepare、sampling、ground solve、DF refit、H4 P0 compile、科学state-actionを呼んでいない。人工PASSをH4性能証拠とはしない。

| 要求項目 | 主な検査 |
|---|---|
| ordered request/q、cut/A/B | C5 request検証、逆順requestのfingerprintと異なるordered operator |
| AB/actual adjoint/vacuum/χ | C5 2-mode end-to-end operator、χ省略negative control |
| scalar/identity、Gaussian/signed tail | 非零scalar/identity一回、number-preserving非可換basis、正負tail、符号binding拒否。既存Stage Aの符号依存χも再利用 |
| 同requestで両方式、X/Y | 共通bundleのwrapper選択、異request拒否、両axis observableとcomplex coherence |
| 境界融合/global cut | q2の1 symmetric境界・Reference内部0・cut損失1、既存C3両control branch・ordinary full検査 |
| SOURCE/endpoint/law | request全identityとcontext source/input不一致拒否、既存Stage A deserialization前拒否 |
| 費用境界/未検証q/未資格 | kernel/shell拒否、raw/paired未PASS拒否、違うP0/axis/fixture/qの拒否、期待値/CI/model価格null |
| C3固定入口 | 保存cc554 SOURCEと新入口を同じ2-mode人工q4/q8で照合しmetadata/operator一致。実H4 q4/q8は再実行しない |
| 保存compileの再利用 | synthetic archiveでraw完全一致時だけcompiled bytesを保全しcompile_calls0。異rawは再利用しない |
| 公開projection | private operational観測を除外し元reportを変更しない |

人工fixtureは実行前にQiskit 1.3.0 gate APIとnumber-form headの真空位相定義を修正した。許容差の緩和・phase fittingは行っていない。Stage A testはunittest discoveryでは必要なbootstrap変数が設定されないため、元SOURCEの正式な`--origin`入口で2件PASSを確認した。過去のtest file自体は変更していない。

再利用した入口は`tests/check_track_d_c3.py`、`tests/check_track_d_c2_q2.py`、`tests/check_track_d_h4_paired_reference.py --origin "$TRACK_D_ORIGIN"`。新検査は`TRACK_D_ORIGIN`を指定して`tests/check_track_d_c5.py`を実行する。環境の絶対path・内部CPU/RAM観測・private logは公開しない。
