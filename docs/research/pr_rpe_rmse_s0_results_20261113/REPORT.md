# H4 DF10 PR–RPE：保存二時点RMSE S0結果（2026-11-13 JST）

**承認された提案A＋S0を一回実行し、m0/R2とm6/R128の経験的nominal Nは各軸64、38となった。** 保存数値gateは両候補で通過した。科学的RMSE適用条件は`UNQUALIFIED_MODEL_ASSUMPTIONS`、十分N・actual N・全RPE RMSE boundは未取得のまま。**新mean/reference/内部対照/normalizer呼出し、H/state array load、sampling、compile、C/Gはすべて0。**

科学実行SOURCEは **`7c3e1e5db0c55dd8249e9030a4c7e9954496fa7e`**。[固定計画](../pr_rpe_rmse_pilot_decision_20261113/EXECUTION_PLAN.md) d97a40ab719bd71fee4ad41d5acc2abe3c89904dの一括scopeだけを実行した。旧提案・templateの未採択/null状態、P1/SOURCE/raw/grant/ledger/marker/KEEP/過去STOPは歴史資料として保全した。ユーザーの実行指示に追加のgrant/承認証明書を要求していない。

[科学結果](run01/result.json)／[条件・A/B全metadata](run01/CONFIGURATION.json)／[B向けN packet](run01/N_HANDOFF.json)／[SOURCE索引](run01/SOURCE_INDEX.json)／[実行出力](EXECUTION.txt)／[HANDOFF](HANDOFF.md)／[人工検証](TESTS.txt)。

## 固定条件とM・κの具体化

Linear H4、隣接1.00Å、STO-3G、保存tol-only 1e−8 DF10、保存weighted-Hermitian completion・fragment順・scalar・H/state/snapshot bindingを保持した。sector Nα=Nβ=2、dimension36。targetは保存DF-Hの同sector ground energy、保存数値stateによる条件付き評価、P=0、無energy shift、保存prior/decoder/profileを保持。数学的な厳密ground同定や元分子HのDF誤差保証を追加していない。

| 項目 | 採択して具体化した値 |
|---|---|
| PR構成 | S2、L_D1、ΔPF=.1、K2、h=.1 |
| 保存λ_1 | **2.8629570945298686**。固定H4 contextからcopy、再計算なし |
| 共通κ_paper | **24.305274577756407**（表示用近似）。`κ=(255/128)/(λ_1²h²)`をexact Fractionで具体化。正本はCONFIGURATIONの長い有理数文字列 |
| 最大round index M | **12**、全metadataは0..12（13round）。承認されたPR経験的horizon式から得た値で、旧M12の無条件copyではない |
| εE | **0.00159360010199040 Ha**を今回はRMSE model targetとして設定 |
| εqpe / εtrot | 各 **0.00079680005099520 Ha**、exactなεE/2 |
| 合成則 / normalization | **minkowski / finite_B_diagnostic**、既存policyを使用 |
| PR経験的horizon尺度 | **0.0007669903939428206 Ha**。既存Cmax=.10のモデル診断であり、実energy RMSEやrigorous boundではない |

κは一つの式を一回具体化しただけで、区間探索・R最適化・別κ試行はない。x=κλ_1²h²=255/128の元整数mapperから、対象二候補のID/構成が保存P1と一致した。Mのmetadata生成は全round信号評価ではない。旧α=.05やβ=.4/.2、旧Hoeffding Nを今回のRMSE Nへ流用していない。

## 対象と保存数値gate

| m | T | qRPE / qPF | R_outer / event数 | physical candidate ID |
|---|---:|---|---|---|
| 0 | .1 | 1 / 1 | 2 / 2 | `53e00d8b07d1547d19a7` |
| 6 | 6.4 | 64 / 64 | 128 / 8192 | `2a8c6718220eec59d419` |

保存P1公開71ba49bb28b295aa28e51101bc3b1bf07d500767（科学SOURCE a1a79213a9b5331145e20e8778a3be9d9fc4e497）の二recordだけを数値再解析した。H4 contextはSOURCE_INDEX 7a8bc7f…の`T20_reference_files.H4`、hash `8396605fa0c5ee34f29b436ee16e0f731cfaa6cd94f3304033106a85e9f5d52f`。実行前後で同一、H/state arrayは読み込んでいない。

元`checked_reference`の二経路gate、保存vector closure/exact-tail norm gate、R-aware operation-count allowance、独立raw/correctedのscalar closure、bias分解の整合性を再確認した。数値allowanceとlogB marginは元値を保持し、閾値変更や不足データのfresh取得は0。

| m | 独立reference差 | 保存vector closure | exact-tail norm誤差 | scalar raw/corrected closure |
|---|---:|---:|---:|---:|
| 0 | 1.110223025e−16 | 1.422233169e−16 | 2.220446049e−16 | 2.775557562e−17 |
| 6 | 5.964543397e−15 | 3.235261818e−13 | 2.220446049e−15 | 3.234598534e−13 |

これは元保存数値証拠の再確認であり、新しい独立伝播の取得ではない。rawをcorrected/Bから作り直していない。原reference/meanはresult内に保存値のまま保持し、元P1 recordは変更していない。

## normalization・bias・phase/radius・nominal N

以下はresultの表示丸め。Bは保存whole logBをexp表示した値で、finite distributionを再計数していない。原D1 diskのscalar式とempirical u/marginを使い、旧Hoeffding wrapperを呼ばず`supplied_disk_geometry`へ渡した。

| m | log B | B | attenuation | \|corrected−reference\| | correctedのreferenceに対する位相 [rad] |
|---|---:|---:|---:|---:|---:|
| 0 | .0404893917899 | 1.04132026310 | .960319351727 | 1.340513015e−5 | −1.340512705e−5 |
| 6 | .0409824941800 | 1.04183386723 | .959845932691 | 8.572682639e−4 | −8.572683121e−4 |

| m | 条件付きraw disk | 条件付きraw半径下界 | βdet [rad] | nominal N_X | nominal N_Y | model status |
|---|---:|---:|---:|---:|---:|---|
| 0 | 1.287435066e−5 | .960306477376 | 1.340632222e−5 | **64** | **64** | UNQUALIFIED_MODEL_ASSUMPTIONS |
| 6 | 8.228468770e−4 | .959023085814 | 8.572698490e−4 | **38** | **38** | UNQUALIFIED_MODEL_ASSUMPTIONS |

Nは指定Mと保存logBを用いた各軸`ceil(exp(2 logB)[11+4(M−m)])`という経験的モデル値。旧高確率型Nとの数値差を、同じ保証の下でのshot改善と呼ばない。旧α/β gateによる十分Nでも、actual minimum Nでもない。

conditional diskは、保存bias/経験的u・保存solver energyに対するreference offset/T×保存residualという従来の数値state条件付きrecipeを維持した。E_stateはこのreference診断にだけ用い、prior/branch選択へ使っていない。ρ/βdetは局所diagnosticで、energyのεtrot上界やRMSE boundではない。actual implementation/law/厳密ground距離はNoneのまま、0として埋めていない。離れた二時点の連続RPE decoderやsimulationは未実行。

## Bへのcoverageと未資格

[N_HANDOFF](run01/N_HANDOFF.json)はB f09の全schedule identity/lawを保持し、m0とm6にだけnominal Nを供給した。**m1..5、m7..12はNOT_EVALUATED、nominal Nはnull。全13roundのN_X/Y（十分N）とglobal RMSE boundはnull。** PF B0 metadataをFull-H PFへrelabelせず、PF/純RTE評価・C/G集計はない。

packetは`status=UNQUALIFIED`、finite_K/signed_piece_kappaはUNQUALIFIED、全round state/target/alias/prior/decoder資格=false。条件採択・実行許可と、error contractの科学的資格は別であり、固定Bの資格fieldを便宜的にEXTERNALLY_QUALIFIEDへ変更していない。

必要な追加証拠は、finite-K平均信号への経験的shot則の移送、state/有効発展モデル・ground/overlap、誤差合成、alias/全round decoder統計、実装/law。今回の二点でこれらを認定しない。M12は決まったが、残りroundの科学評価を許可するものではない。

## 実行範囲・検証・STOP

| 処理 | 実施 |
|---|---:|
| 固定(M,κ)の具体化 | 1組 |
| 保存二候補のgate・局所診断 | 2候補 |
| 経験的nominal N | 2候補×X/Y |
| 新mean / 新内部対照 / 新reference action / finite normalizer | **0 / 0 / 0 / 0** |
| H/state array load / sampling / compile / C/G | **すべて0** |
| 科学retry・候補差替え・閾値変更 | **0** |

薄い固定S0 runnerを追加し、既存model/reference checker/geometry/B packetを再利用した。人工7テストPASS後にSOURCEをcommit固定し、実S0は一回だけ実行。[TESTS](TESTS.txt)参照。消費した元SOURCEファイルhash、context前後一致、13round packetのidentity/law・null、旧template不変を確認した。作者自身の技術・数値確認であり、独立科学レビューではない。

**Codex B**へは[HANDOFF](HANDOFF.md)と未資格packetを公開する。**ユーザー**は次のRMSE適用条件・全round検証の有限scopeを判断・承認し、**Codex A**は別途承認後だけ実施する。今回はcommit/push・remote SHA/主要成果物照合後**mandatory STOP**。S1不足取得、他round、H6、P2/P3、価格・全G、詳細科学レビューへ進まない。
