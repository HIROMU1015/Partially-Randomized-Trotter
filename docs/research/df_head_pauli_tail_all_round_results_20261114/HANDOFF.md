# A → B/C/D：Pauli LD1/LD2全13round signal

新科学SOURCE `7084fbc9a43cf7b7e24681b4005d37cb59c1e465`、新raw carrier `fce4e4f7eae620e4440ac4103312d7afed9e9608`。旧4件はP1 `e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7`、数値SOURCE `05447d1d4f1e3c0e9e335a7085aa58a877c139f2`から再計算せず再利用。P0辞書/operatorは4220f3006、B validatorはP1と同じce0517de。[REPORT](REPORT.md)／[全mean packet](MEAN_HANDOFF.json)／[全26row CSV](ROUND_SUMMARY.csv)／[SOURCE](SOURCE_INDEX.json)。

## Canonicalとcoverage

LD1 ID `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27`、LD2 ID `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3`はP1から不変。DF IDの流用、辞書の再採択、κ/Rの再探索はない。

各packetのconfiguration/13round identity/lawを正本とし、signal_coverage_complete=true、evaluated_rounds=0…12。reused_rounds=[0,6]、new_rounds=[1,2,3,4,5,7,8,9,10,11,12]を分離した。全26rowのN/nominal N/十分N/global RMSE/C/Gはnull。全signal取得済みという状態はRPE全体の精度資格を意味しない。

completion scientific sourceは新assemblyを指す。各rowのmean_sourceとmean_record.sourceが数値の正本SOURCEである。旧4recordはSOURCEを含めそのまま保持し、locatorをP1固定commit/pathへ接続した。新22rowはfce4e4f7eの個別JSONを参照する。P1のfixed-R/full256 controlはSOURCE参照だけを残し、共通κへ混ぜていない。

## 物理・数値契約

保存H4 DF10/Linear1Å/STO3G/同H/state/sector(2,2)36、S2/Δ.1/K2/h.1/M12/元exactκ・target/prior/decoder/誤差budget・scalar一回を維持。Pauli辞書identity/law/λ/bit順はP0正本と対応する。今回の新22件はsector36 ideal meanであり、新full256 mean/referenceは0。

- rawを理想Hadamard X/Yのp=(1+Re/Im raw)/2へ渡す。Bやcorrectedを確率へ二重適用しない。
- raw/correctedは独立伝播。closure最大4.3729e−10は固定1e−9 gate内。exact-tailと保存state full-H referenceは別入力。
- normalizer丸め感度とR-aware u/logB marginは各rowにある。m12のuはLD1約2.5361e−6、LD2約9.8005e−7で、rigorous上界ではない。signal差をTで割ってenergy RMSEへ変換しない。
- P0 full256 operator検証/P1一対照をSOURCE付きで保持。個別Pauli eventはsector外へ出得るので、射影Pauliを36次元unitaryとして実samplerへ使わない。
- 厳密ground同定・actual sampler/control法則・全decoder RMSE移送は未資格。unknownを0に置いて保証へ昇格しない。B riskが含むPF/finite-K効果へ別δE等を二重加算しない。

## C/Dと次段階

価格担当には同configuration/candidate ID、全round T/qPF/qRPE/R_outer/allocation/K、dictionary source/hash/λ/scalar、head/fragment順、control/compiler/profile/P=0を渡す。m12のlogical event位置はLD1 44,584,960、LD2 17,223,680であり、compiled RZ数ではない。今回巨大event列・回路・価格は生成していない。

C/Dはwhole wrapper/lowering・Pauli support/basis/product/order phase・head境界と同lawを既存pipelineへ接続する。旧DF/P3/O1価格を新候補の平均Cへ移送しない。Gは適格な全round N/Cが別途揃ってから扱う。

Codex B：公開packetのidentity/SOURCE/mean受入を確認する。
ユーザー：N/MSE・価格等の次の科学実行scopeを判断・承認する。
Codex A：今回の全signal取得を終了し、公開・remote照合後mandatory STOP。
