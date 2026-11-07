# 次に提案する有限の技術確認

2026-11-08。**案Mの共通factorizationと測定instrumentを先に確認することを提案する。** 今回は設計でSTOPし、以下の実装・テストを開始していない。候補の採択・新規性認定・科学的GOではない。

## 優先する理由と到達目標

[案M](O2_NEW_ALGORITHM_CANDIDATES.md)は、metadata CVでは削れなかった量子readoutの情報を利用し、LCUのherald/成功率補正を必要としない。群内labelを別々に実装する代わりに平均observableを測るという操作差がある。同じ条件付き平均とM2≤1を記号的に示せる一方、full価格や実H4適合は未知である。最初に知るべきことは「実験で相関が良いか」ではなく、**元selected targetのまま群を作れて、その測定を明示できるか**である。

## 次回、許可を得た場合の最小技術scope案

1. 保存metadataのschemaとselected-basis SOURCEを読み、群適格性の必要十分な検査条件を固定する。1 slotに限定し、同一のA/B/V/φ、signed D_l、phase、qubit orderingとactual qの定義を示す。近似行列一致や別basisの名称一致では受理しない。original basisへ置き戻す場合も、selected full targetとの誤差を勝手に増やさない。
2. **実H4 payload・stateを読み込まない**最小adapterを別SOURCEで作る場合は、既存のsystem gate IR、Hadamard規約、bit-only facadeへ枝配置・system readout・重みdecoderだけを追加する。共通frameworkや全slot探索は作らない。RNGやcompiled価格取得はまだ不要。
3. 次回明示的に許可された範囲で、解析解がある小規模fixtureに限定してinstrument correctnessを確認する。異なる対角supportの群、rotation/product、既知event phase、X/Y符号、逆gate順、identity、m=0、単独label、非共通basis拒否を対象にする。有限個の全結果について平均とM2の式を照合し、旧結果や実stateを使わない。これは今回の禁止範囲にある新しい量子測定simulationを開始する許可ではない。
4. finite-grid群質量の整数保存、重み丸め、軸生成/測定確率/bit-gridの誤差項を列挙する。旧b_engineでは足りないことを明記し、将来保証できる範囲とUNKNOWNを分ける。未知項を0にしない。
5. full-wrapper資源表はこの段階では記号的に作る。既知単独label direct測定とgroupingをbaselineに含め、A†/B制御、basis、ancilla軸、system readout/feed-forwardと古典r(z)の追加を表示する。実compileやN評価へ自動移行しない。

判別の成果は一つのadapterが動いたというだけでなく、target一致の条件、実装可能なreadout、未知の新gate誤差、比較すべき強いbaselineが固定されることである。条件を満たさなければ案Mの対象classを限定するか保留する。Dへ自動的に切り替えて実装・科学実行しない。

## 科学入力へ進む前の別の有限計画

実H4へのgroup eligibility適用、実Qの新しい群質量/r(z)生成、実stateでのgate/Born/測定、価格取得、母分散/十分N/総費用評価は、ここまでの技術確認とは別の科学scopeにする。N1/N2の再solve/preparationや旧認証の再実行は不要であり、禁止を維持する。

将来は、先にmetadata-onlyで**1 slotの有限な候補群**と元target対応を固定し、それが通った場合だけ、独立した新law/stream/source/outputで一条件・二方式の有限測定案を設計する。最初から18roundや4方式・保証付き総費用比較を要求しない。shot数やε/α、seedは今回設定しない。旧752/SCVのshot数・係数・bit・cacheも引き継がない。

実科学の比較では、元selected Q、保存linear H4／1.00 Å／STO-3G／legacy DF12／L_D=3／m0 q1 δ=0.02 r1 K2／coefficient_atol=0.0と全4電子sector groundを使うかを明示的に採択し、旧engine保証と新instrument保証の違いを埋める必要がある。同じ誤差・失敗条件に対するrange/variance、全段階のfull compiled費用・readout・古典負担を評価する。単一campaignの分散/費用傾向は期待費用改善の証明にしない。

## STOPと次に必要な判断

共通factorization不成立、scalar/phase不一致、正support欠落、actual確率と重みの不一致、qubit orderingやdecoderの符号不一致、新誤差項の未保証があれば、条件を緩めずその境界でSTOPする。実wrapperの価格が不明なのに局所gate削減だけで実行GOとしない。

次のユーザー判断は案Mの**限定技術実装・人工instrument検証を行うか**である。新科学計算はその結果と有限契約を別途確認して明示的に許可する。重要な新理論・新規性・研究方針の採択を詳細科学レビューで判断する場合も、今回の候補提案とは別の開始指示を要する。承認証明/grant発行の新制度は作らない。

案Dは、coarse入力元・補正全費用・既知角度補間/C-Sとの違いを先に整理できた場合の後順位案として残す。小さい局所sin係数だけを根拠に448/752-shot等の実験を組まず、C-SやO3の固定SOURCE/STOPを維持する。
