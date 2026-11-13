# A P1 → B/C/D：Pauli二時点mean

科学SOURCE **05447d1d4f1e3c0e9e335a7085aa58a877c139f2**、P0 **4220f3006d0d8d413e195ed948e250578513b557**、B **ce0517de6e7b58cf4bf4c69f99f998d86099c406**。保存H4 DF10/Linear1Å/STO3G/sector(2,2)、S2/Δ.1/K2/h.1/M12/元exactκ、同H/state/target/scalar。6 mean/18complex・normalizer5unique、全局所数値gate通過。[REPORT](REPORT.md)／[機械可読mean packet](MEAN_HANDOFF.json)／[全plan](run01/PLAN.json)。

## B canonicalとpacket

Pauli共通κ configuration：LD1 `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27`、LD2 `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3`。Bのnative generate/_validateで辞書付きidentityを生成した。元DF canonical IDは比較元として元SOURCEで保持し、新Pauli IDへ流用していない。

packet内のcanonical configuration/13round identity/lawを正本にする。各構成で取得済みはm0/m6だけ。残り11roundのmean・全roundのnominal/十分N・global RMSE/C/Gはnullである。これはN packetではなくmean evidenceである。mean_sourceのpath/selectorはこの結果release内のcarrierを指し、scientific_sourceは上記05447d1d4を指す。

fixed-R対照ID `81afc1a8c0660b116ac656b6411ea01028a0a56f474aa0b3ba664d0ce5ae3126`のm0/R2と、そのfull256 meanはcontrol_evidenceへ分離した。元κ値は保持するが、このcontrol roundではκ適用=falseである。controlを共通κのm0/R3として取り込まない。他roundのcontrol metadataを科学取得済みとしない。

controlの`event_mapping.source`は基礎となるB paper mapperを指す。m0だけのoverride実装はAの上記科学SOURCE・`df_head_pauli_tail_p1.generate_configuration`であり、packetのcontrol_evidence.provenanceで両者を明記した。既存canonical配置・IDを後から変更していない。このA診断対照をBが独自に採択した科学policyとは呼ばない。

新Pauli R_outerはLD1 m0/m6=3/171、LD2=2/66。旧DFの2/128、1/40とは異なるので同条件signal差は算出していない。LD1/m0の固定R2だけはq/δ/K/head/scalar/配分を確認して比較し、corrected差2.7756e−17、raw差.01269336は辞書依存として記録した。

## 信号・law・数値の意味

- `record.signal.raw`を理想Hadamard X/Yのp=(1+Re/Im raw)/2へ渡す。correctedやBを確率へ二重適用しない。wrapper/control/native実装の資格は今回取得していない。
- raw/correctedは独立に伝播し、exact-tailはunitary内部対照、referenceは同saved-state full-Hの保存独立pairである。各differenceは無次元signal空間で、Tで割ってenergy bias/RMSEへ置き換えない。
- λ_Pとdictionary source/hash/bit順/scalarをsplit/law/contextに含めた。辞書hashはP0正本と一致。計算tailはP0 Pauli非identity和、headは元block、保存s_Rとcは一回適用。P0のs_P−s_R/丸め残差を無断補正しない。
- 36次元は合計ideal mean用である。個別Pauli eventはsectorを出得るため、実sampler/native eventを射影Pauliのunitaryとして実装しない。full/sector meanの数値照合はLD1/m0/R2の1点だけ。
- allowanceは従来のR-aware empirical policy。数値gate PASSをrigorousな誤差上界やground/RMSE資格へ昇格しない。unknownなground同定・actual sampling law誤差は未取得のまま。

Bは既存controller/risk/decoder/shot処理を再利用して、このpartial evidenceを接続できる。今回Aはそれらを呼んでいない。B riskが含むPF/finite-K biasへ別のδE等を二重加算しない。

## C/Dと次の承認

価格担当には同canonical configuration、全13round metadata、P0 dictionary source/hash、R/q/allocation/Kとhead/scalar/control profileを渡す。Pauli primitiveのsupport/basis/product/order phaseとDF head境界を含むnative/lowering接続は別scopeで確認する。旧DF価格や旧P3/O1価格を新Pauliの平均Cとして流用しない。

ユーザー：次の対象prefix・不足roundの有限科学scopeを判断・承認する。
Codex A：承認されたsignal範囲だけを実行する。
Codex B/C/D：各担当の受入/統計/価格条件を別の承認・作業範囲で扱う。

P1科学実行は完了。公開・remote照合後mandatory STOP。全13round signal、N/MSE、compile/価格/Gや最適化は未実行。
