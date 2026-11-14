# 5方式の物理候補と保存信号の対応、Full PFの全13round原本調査

2026-11-15 JST。担当はA（主agent Claude Code）。

**作業の範囲**
- 保存資料・SOURCEを照合し、metadataを処理しただけである。
- 新しい信号・固有系・平均の算出、保存スペクトルからの評価、MSE・N・G、回路のbuild・compileは0。
- 保存値の比較は、同じ物理candidateであることの確認だけに使った。
- 作業は作者自身によるもので、独立した査読ではない。

**成果物**

| file | 内容 |
|---|---|
| [SIGNAL_COVERAGE.csv](SIGNAL_COVERAGE.csv) | 候補×round×X/Y、364行 |
| [SAVED_ROUND_AUDIT.json](SAVED_ROUND_AUDIT.json) | 決定論PFの保存round meanの索引。710 record |
| [BUILD_SIGNAL_COVERAGE.py](BUILD_SIGNAL_COVERAGE.py) | 固定blobだけを読み、`--check`で不変条件を検査する |
| [HANDOFF_B.md](HANDOFF_B.md)、[HANDOFF_C.md](HANDOFF_C.md) | B/Cへの引継ぎ |
| [TESTS.txt](TESTS.txt) | 検査結果 |

**既存資料の再利用**：新しい形式は、上記の結合だけにとどめた。次の既存資料をそのまま読んでいる。

| 資料 | 場所 |
|---|---|
| BのREADINESS・SIGNAL_APPLICABILITY | `99930491` |
| A follow-upのC_HANDOFF・ROUND_SCHEDULE | `714247619` |
| T2のCONFIGURATIONS・cell | `5fb2c32e` |
| CのCELL_PLAN_V2・PR_DF_COVERAGE | `855573064` |
| Cの39round manifest・COVERAGE_AND_REUSE | `c5466946` |

## 0. 結論

**全13roundの信号packetが揃っている候補**
- PR-Pauli LD1/LD2（`d905d219`／`d7dd4f52`）
- Full-H κ2（`54ac3eb3`）
- Truncated B0I p7・S2・幅0.1（`e6754564`）
- 以上はBのmetadata intakeがPASSしている。
- Full-H R448（`cfa6aee3`）とPR-DF LD1/LD2（`a4dda8f7`／`1fa33556`）も、13 rawは保存済みである。
  - R448は比較候補ではなく、intakeの記録もない。
  - PR-DFは、Bの版付きdecoder-only bridgeを待っている。

**Full PF**
- 確認した資料の範囲に、正式な全13round packetは**一つもない**（S2・幅0.1も、shortlistの3候補も同じ）。
- これは「過去に未実施」という意味ではない。Cの39round計画（`c5466946`）にはS2・幅0.1の13roundが含まれていたが、承認scope（`6306fb3a`）はB0I・R448だけを実行した。
- 保存meanがあるroundは、shortlistの各候補で13roundのうち5〜6 roundである。
  - 現行SOURCEの二時点direct meanが4〜6 round。
  - Track Aの旧meanが1 round。

**Truncated PF**
- 信号が揃っている候補（S2・幅0.1）と、価格を取得した・取得予定の候補（S2・16h/7、Y4・16h/3、M5・16h）が**別の物理候補**であることを、ID単位で確認した。
- shortlistの信号は、Full PFと同じ構成である。保存があるのはT2の二時点meanとTrack Aの旧meanだけで、各候補のm5〜m11は保存meanがない。

**精度側と価格側のIDの一致**
- 価格cellがあるすべてのroundで、精度側と価格側のIDが一致した。
- 決定論PFは、configuration ID・physical IDとも一致した。
- PR-DFは、26round×X/Yのphysical IDが一致した。
- PR-Pauli LD1 m0とFull-H κ2 m3は、identityとlaw_bindingが完全に一致した。
- 不一致は0件である。

**最小の追加計算**（§7、未承認）

| 対象 | 新しいround mean |
|---|---|
| Full PF shortlist | 22件（m3の旧meanを使わない場合は24件） |
| B0I p7 shortlist | 同数 |
| Full PF S2・幅0.1を候補に残す場合 | 8件（旧値を再資格化しない場合は最大11件） |

- 各stepの固有系・演算子はすでに保存されている。
- ただし、平均の算出は新しい科学計算であり、今回は行っていない。

## 1. 照合した最新状態（開始時）

| branch | HEAD | 内容 |
|---|---|---|
| `codex/research-roadmap-integration-20261114` | `07a6e0095` | 指示の`fa84c0d6`より後。§10.6を追記している |
| `codex/rpe-total-dp-acceleration-20261114` | `999304918` | Bの適用表 |
| `codex/m5-m4-receipt-admission-20261114` | `855573064` | C |
| `codex/d-price-records-c-bridge-20261114` | `acea1118a` | D |
| `codex/pr-r-dependent-n-model-20261111` | `714247619` | Aの信号 |
| A作業branch | `7a0c9468b` | 本作業でも継続して使用 |

- 指示に書かれたcommitより新しいHEADは、roadmap以外にない。
- Aの信号branchは、別のworktreeでcheckout中である。そのため変更せず、固定blobを読むだけにした。

## 2. 物理候補の同一性

**全候補に共通する条件**（各packetのbinding・INPUT_PROVENANCEで確認）

| 項目 | 値 |
|---|---|
| H | `c1a6c428…` |
| snapshot | `12219b7d…` |
| state | `9bcc0b0e…` |
| 系 | Linear H4、1.00 Å、STO-3G、tol-only 1e−8 DF10、sector(2,2)/36 |
| fragment順 | 元の順序 |
| reference | 保存reference pair（`bba15083`の13時刻） |

**候補ごとの条件**

| 候補 | 保持・表現 | 公式・schedule | 幅・q | R・K・κ・law | scalar | configuration ID |
|---|---|---|---|---|---|---|
| PR-DF LD1/LD2 | DF headはprefix 1/2、DF tailはRTE | S2、signed weight [1] | 幅0.1、q=2^m | Rは元κのgrid（LD1：2…8160）、K2、law `ideal_finite_RTE_law`（DF component） | 総Tで一回（`scalar_identity_once_at_total_T`） | `a4dda8f7`／`1fa33556` |
| PR-Pauli LD1/LD2 | DF head＋**Pauli辞書184項**のtail（DF tailではない） | S2、[1] | 幅0.1、q=2^m | R（LD1：3…10885）、K2、元のexact κ（PAPER_TOTAL_ROTATIONS）、law（Pauli辞書） | c＋保存s_Rを一回（`KEEP_SAVED_sR_AND_c_ONCE`） | `d905d219`／`d7dd4f52` |
| Full-H κ2 | 一体項を含む全H。DF headなし | 純RTE（L_D=0） | groupはq=2^m、T_m=0.1·2^m | R=9…36223、K2、κ=2、law（Full-H成分） | c＋抽出identity（−0.33147781…）を一回 | `54ac3eb3` |
| Full-H R448 | 同上 | 同上 | 同上 | **R=448固定**、κなし | 同上 | `cfa6aee3`（κ2とは別） |
| Truncated B0I p7・S2・0.1 | prefix 7を保持し、tail identityをscalarへ（B0I） | S2、[1] | 幅0.1、q=2^m | 決定論（B=1） | c＋B0I tail identityを一回 | `e6754564` |
| Truncated B0I p7 shortlist | 同上 | S2/Y4/M5 | 16h/7・16h/3・16h、早期roundは例外の幅（§3） | 決定論 | 同上 | `a6e55777`／`defc5e88`／`424eecbf` |
| Full PF S2・0.1 | 全10 fragment＋一体項 | S2 | 幅0.1、q=2^m | 決定論 | cを一回 | `f2d58f32`（C manifest） |
| Full PF shortlist | 同上 | S2/Y4/M5 | 16h/7・16h/3・16h、早期roundは例外の幅 | 決定論 | cを一回 | `11cf8ca6`／`a799697f`／`70f7b317` |

**physical IDの作り方**
- Track Aの`cid`（system、method、formula、prefix、T、q、R、K）を使う。
- 保存recordが持つIDは、どれも自身のfieldから再計算した値と一致した（builderの検査）。
- PR-Pauliでは、canonical ID＝physical ID（Cの辞書bridge）である。
- Full-Hのphysical IDは`FULL_H_`で始まる。

**信号field**
- 13round packet（PR-DF、PR-Pauli、κ2、R448、B0I p7・0.1）は、raw・corrected・reference・normalization・uncertainty・qualificationを持つ。
- Full PFとB0I shortlistの保存meanは、direct PF meanだけを持つ。B=1で、raw＝correctedである。
  - referenceは保存reference pairから束縛する必要がある（metadata bridge）。

## 3. Full PFの全13round調査（最優先）

**調査した資料**

| 資料 | 内容 |
|---|---|
| A S2/.1 H_eff pilot（`69ee0ff8`） | m0・m12のdirect mean |
| A H_eff scan（`3ae055fa`→`714247619`） | 14 cell×二時点 |
| A follow-up（`a3005d72`→`714247619`） | 5 cell×二時点 |
| C 39round計画（`c5466946`） | S2/.1の計画と、保存coverageの判定 |
| 4方式screen run02（`699178305`） | m6を4 cell保存。m12は数値gate失敗。run全体はSTOP |
| Track A長時間 signal_records（`58b21d596`） | T∈{0.8, 1.2, 1.6, 2.0}のFULL/B0I |
| Track A pf_common H4（`0b1886f53`） | T=0.8 |
| B適用表（`99930491`） | |
| baseline pipeline | 生成器のみで、保存catalogではない |

saved round meanを710件索引し、physical IDで照合した。

**判定区分**

| 区分 | 意味 |
|---|---|
| S1 | 保存packetをそのまま受入できる |
| S2 | 現行SOURCEの保存direct meanがある（数値gate PASS）。packet化とBのintake照合が必要 |
| S2L | 旧SOURCEのmeanがある。physical ID・入力は一致するが、現行のpower診断がない、または旧run全体がSTOP。再資格化が必要 |
| S3 | stepの固有系・演算子は保存済み。新しいmeanの算出が必要（未実行） |

**Full PFの各候補×roundの状態**

| m | S2・16h/7 | Y4・16h/3 | M5・16h | S2・0.1 |
|---:|---|---|---|---|
| 0 | S2（pilot、2nd@1h） | S2（scan、4th@1h） | S2（scan） | S2（pilot） |
| 1 | S2（scan、2nd@2h q1） | S2（scan） | S2（scan） | S3 |
| 2 | **S3**（2nd@2h q2） | S2（scan、4th@4h q1） | S2（scan） | S3 |
| 3 | S2L（Track A、q4） | S2L（Track A、q2） | S2（scan M5@8h）＋Track A。差4.5e−13以下 | S2L（Track A q8） |
| 4 | S2（follow-up q7） | S2（follow-up q3） | S2（follow-up）＋Track A。差4.5e−13 | S2L（Track A q16） |
| 5〜11 | **S3**（2nd@16/7h） | **S3**（4th@16/3h） | **S3**（M5@16h） | m6はS2L（screen F2-FINE）、他はS3 |
| 12 | S2（follow-up q1792） | S2（follow-up q768） | S2（follow-up q256） | S2（pilot q4096。旧spectralのgate失敗を、direct経路でPASSしている） |

**読み取れること**
- H_effの固有値shiftと二時点の照合は、全roundのraw信号の代わりにならない（B適用表と同じ判断）。
- S2・16h/7のm2・m3は2nd@2hのstepを使う。m3はTrack Aにしか保存がない。
- Track Aの2資料は独立ではない。長時間studyはpf_commonの値をT=0.8の資産として再利用しているので、両者の一致（差0）は独立の照合ではない。
- 独立の照合になっているのは、M5のm3・m4で現行SOURCEの値と旧値の差が≤4.5e−13だった点である。これは保存値どうしの照合であり、新しい計算ではない。
- M5・16hのspectralな経路では、承認済みのENERGY_WINDOWを使う必要がある（follow-upと同じ）。direct powerならwindowは不要である。
- **S2・0.1の扱い**：Cは以前（`c5466946`）、Track A旧値のm3・m4を「物理は一致するが、現在のpower診断が欠けているので新規directで評価」とし、screenのm6を「受入再資格の1件」としていた。本調査はその判定を保持する。その後に出たA pilot（`69ee0ff8`）が、m0とm12の現行direct meanを加えた。

## 4. Truncated PF shortlistの信号coverage

| m | B0I p7 S2・16h/7 | Y4・16h/3 | M5・16h | 既存のS2・0.1 packet |
|---:|---|---|---|---|
| 0 | S2（T2、packetのm0と同じ物理candidateで差0） | S2（T2） | S2（T2） | S1 |
| 1 | S2（T2） | S2（T2） | S2（T2） | S1 |
| 2 | **S3** | S2（T2） | S2（T2） | S1 |
| 3 | S2L（Track A） | S2L（Track A） | S2（T2）＋Track A。差1.4e−14 | S1 |
| 4 | S2（T2） | S2（T2） | S2（T2）＋Track A。差4.8e−13 | S1 |
| 5〜11 | **S3** | **S3** | **S3** | S1 |
| 12 | S2（T2） | S2（T2） | S2（T2） | S1 |

- T2の二時点meanは`check_only_not_in_any_fixed_schedule=false`、数値gateはPASSしている。
- T1/T2の切断energyとH_effの誤差は、全RPE信号・MSEの代わりにしない。
- 価格がある、または予定されているのはshortlistのm4〜m6だけである（M5のm4は受入済み、他は較正またはholdoutとして取得待ち）。
- 一方、信号packetが揃っているのはS2・0.1である。どちらの候補に揃えるかは、ユーザーが判断する。

## 5. PR-DF・PR-Pauli・Full-H

| 系列 | 同じphysical contextか | law・normalization・scalar・幅 | decoder viewとの対応 | 不足しているもの |
|---|---|---|---|---|
| PR-DF LD1/LD2 | 26round×X/Yのphysical IDがCのPR_DF_COVERAGEと一致。D P2 pilotのm0（q1・R2）・m6（q64・R128）も同じID | 信号はideal finite law、K2、元κ、幅0.1。価格側のscalar SOURCE bridge（c＋s_Rの訂正、D `e8273676`）はCの受入待ち | 元ID→total IDは未生成（`A_FIXED_GRID_510FBACD_V1`がT.createで拒否される） | **metadata bridge**（B：版付きdecoder-only bridge）と、価格の受入。新しい信号は不要 |
| PR-Pauli LD1/LD2 | LD1 m0の価格（B0）とidentity・law_bindingが完全一致。LD2は価格がない | 辞書のSHA、λ_P、s_R、bit順はpacketに束縛済み。DF tailと混同していない | 元ID（＝physical ID）とtotal viewはBで束縛済み（eecbc→T.create） | 価格（B1）。信号の不足はない |
| Full-H κ2 | m3のP-1価格とidentity・law_bindingが完全一致 | κ2、K2、finite normalizer、scalar一回 | 元ID `54ac3eb3`→total `265b90dc`（B） | 価格（P-2）と、科学的な資格（risk契約・sampler・ground） |
| Full-H R448 | 価格なし | κなし、R448 | intakeの記録なし | 比較候補ではない。κ2と結合しない |

**最終G比較への移送について**
- どの系列も、保存数値の資格は「保存state・ideal law・binary64の診断」に条件付いている。
- 実samplerへの移送、ground認証、数値certificateは未取得である。
- 最終比較へ移送するには、Bの契約採択と、同じclaim水準での照合が必要である。

## 6. 保存結果を再利用できるかと、その根拠

| 保存結果 | 判定 | 根拠SOURCE |
|---|---|---|
| PR-Pauli 26 mean、κ2 13、B0I p7・0.1 13 | そのまま受入可（S1） | `714247619`、`ce6aacfa`、`5fb2c32e`。Bのintake（`99930491`） |
| PR-DF 26 raw | 信号は再利用。bridge待ち | `714247619`（A91349のcopy、照合済み）。Bの判定 |
| Full PF・B0Iの二時点direct mean（S2） | 再利用候補。packet化（reference・identity・law・B=1）とBのintakeが必要 | `69ee0ff8`、`714247619`（scan・follow-up）、`5fb2c32e`（T2） |
| Track Aの旧mean（S2L） | 条件付き。現行のpower診断がないので再資格化するか、新規directにする（Cの前例） | `58b21d596`、`0b1886f53` |
| screen run02のm6（S2L） | 条件付き。run全体はSTOP。m6の元gateはPASSで、Cの「受入再資格1件」の対象 | `699178305` |
| H_effのshift・T1/T2の切断energy | 信号の代わりにならない | B適用表 |
| R448 | κ2・価格へ流用しない | B適用表、roadmap §5 |

## 7. 追加の科学計算が必要になる最小一覧（未承認・未実行）

**必要な件数**

| 候補 | 新しいround mean（S3） | 条件付き（S2Lを使わない場合） |
|---|---|---|
| Full PF S2・16h/7 | m2、m5〜m11（8） | m3 |
| Full PF Y4・16h/3 | m5〜m11（7） | m3 |
| Full PF M5・16h | m5〜m11（7） | — |
| B0I p7 S2・16h/7 | m2、m5〜m11（8） | m3 |
| B0I p7 Y4・16h/3 | m5〜m11（7） | m3 |
| B0I p7 M5・16h | m5〜m11（7） | — |
| （任意）Full PF S2・0.1 | m1、m2、m5、m7〜m11（8） | m3、m4、m6 |

**計算の性質**
- 各stepの固有系・演算子は保存済み。Full PFは`69ee0ff8`・`3ae055fa`・`a3005d72`、B0Iは`5fb2c32e`のT2 cellにある。
- 新しいmeanは、既存のdirect power経路（36次元）で作れる。
- ただし保存スペクトルを使う場合でも新しい科学計算になるので、承認が必要である。
- 既存のgate（power unitarity、norm、二時点照合）をそのまま使う。

**計算以外に必要なこと（新しい信号ではない）**
- S2 roundのpacket化：保存reference pair、identity・law_binding、B=1を束縛する。
- PR-DFの版付きbridge。

## 8. 次の科学実行で得られる研究上の判断

- **決定論PFの同一候補比較が可能になる**
  - Full PFとB0I p7のshortlist（6候補×7〜8 round＋条件付き）を計算すれば、価格側（CのCELL_PLAN_V2、M5 m4受入済み）と同じ物理候補で、全13roundの信号が揃う。
  - その結果、Bが同一候補のMSE・Nを、Dが価格を出し、決定論PFのGを初めて同一候補で構成できる。
  - Truncatedの候補不一致も解消する。
- **H_effの局所診断と全round信号の関係を確認できる**
  - H_effの局所予算判定（内・超過）が、全roundの信号bias・decoderの観点でも同じ順序を保つかを確認できる。
  - これは比較ではなく、入力の整備である。
- **代替案**：S2・0.1側に揃える場合は、価格の取得計画（C/D）を新たに立てる必要がある。

## 9. ロードマップへの影響（§10.6。本体は統合担当が更新する）

| 箇所 | 更新すべき内容 |
|---|---|
| §4.2の到達点と不足 | 「Full PFの全13round packetは未同定」を、round単位の状態に置き換える。shortlistはS2が4〜6、S2Lが1、S3が7〜8。S2・0.1はS2が2、S2Lが3、S3が8。S2・0.1は「未実施」ではなく、Cの39round計画の未承認部分である |
| §5の表 | Full PF shortlistとB0I p7 shortlistの「信号」欄を「m0〜m4・m12に保存mean（一部は旧SOURCE）、m5〜m11は未取得（保存スペクトルから計算可能、承認待ち）」とする。精度側と価格側のIDが一致したことを記す |
| §7の課題 | 「Full PF全13round信号」と「候補の対応付け」は照合を完了した。残るのは最小22（＋2）件×2方式の計算承認と、S2 roundのpacket化 |
| 依存関係 | 決定論PFのG構成は、上の計算（A）→Bのrisk・N→Cのmodelに依存する。PR-DFは、Bのbridgeだけで信号の不足はない |
| 承認 | 必要（新しいround meanの算出）。照合の完了は実行許可ではない |

## 10. 保全とSTOP

- 科学SOURCE・raw・旧数値結果・grant・ledger・marker・KEEP・STOPは変更していない。
- Aの信号worktree、B・C・Dのbranch・processには触れていない。
- Track Aのgz原本は、履歴のcommitからread-onlyで読んだ。
- **mandatory STOP**：不足信号の新規計算、数値の再解析、RPE risk、N、Gへは進まない。
