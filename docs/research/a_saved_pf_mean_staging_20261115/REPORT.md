# Full PF・Truncated PF（B0I p7）shortlist：保存meanのpacket化と再利用の準備

2026-11-15 JST。担当はA（主agent Claude Code）。

**作業の範囲**
- Aの照合結果（`c212cee85`の`a_signal_candidate_identity_coverage_20261114/`）を引き継いで行った。調査はやり直していない。
- 保存SOURCEの読取、metadataの結合、packet化、Bの既存intakeとの技術的な接続確認、人工・回帰testだけを行った。
- 新しいmean・spectrum・power・gate・bias・allowanceの算出、数値の再解析、MSE・N・G、compileは0。
- 作業は作者自身によるもので、独立した査読ではない。

**成果物**

| file | 内容 |
|---|---|
| `STAGED_SIGNAL_INPUT_<候補>.json`（6候補） | Bのpacketと同じ形の段階packet。**未完成** |
| [STAGING.json](STAGING.json)、[STAGING_STATUS.csv](STAGING_STATUS.csv) | 候補×round×X/Yの状態 |
| [S2L_AUDIT.json](S2L_AUDIT.json) | S2Lの技術調査 |
| [B_INTERFACE_CHECK.json](B_INTERFACE_CHECK.json) | Bの既存intakeとの接続確認 |
| [TESTS.txt](TESTS.txt) | test結果 |
| [HANDOFF_C.md](HANDOFF_C.md)、[HANDOFF_B.md](HANDOFF_B.md) | C/Bへの引継ぎ |

**実装**（staging SOURCE `3d64d6891`）
- [`saved_pf_mean_staging.py`](../../../src/trottertracks/saved_pf_mean_staging.py)
- [`stage_saved_pf_means.py`](../../../scripts/stage_saved_pf_means.py)
- [`check_staged_pf_b_interface.py`](../../../scripts/check_staged_pf_b_interface.py)
- test：[`check_saved_pf_mean_staging.py`](../../../tests/check_saved_pf_mean_staging.py)

出力はcommit `bb5fe14ad`で公開した。

## 0. 結論

**6候補の状態**

| 候補（configuration） | 受入準備が済んだround（S2） | S2L（資格確認待ち） | S3（新しいmeanが必要） |
|---|---|---|---|
| Full PF S2・16h/7（`11cf8ca6`） | m0、m1、m4、m12 | m3 | m2、m5〜m11（8） |
| Full PF Y4・16h/3（`a799697f`） | m0、m1、m2、m4、m12 | m3 | m5〜m11（7） |
| Full PF M5・16h（`70f7b317`） | m0〜m4、m12 | — | m5〜m11（7） |
| B0I p7 S2・16h/7（`a6e55777`） | m0、m1、m4、m12 | m3 | m2、m5〜m11（8） |
| B0I p7 Y4・16h/3（`defc5e88`） | m0、m1、m2、m4、m12 | m3 | m5〜m11（7） |
| B0I p7 M5・16h（`424eecbf`） | m0〜m4、m12 | — | m5〜m11（7） |

- 合計はS2が30 round、S2Lが4、S3が44。metadataの不備が残るroundは0。
- **6候補とも全13roundは揃っていない。** 段階packetは`require_complete`で拒否され、Bの`T.bind_signal`でも拒否された。どちらもrisk・N入力にはならない。
- **追加取得の件数は前回から変わらない。** S3が44件。S2Lを使わない場合は最大48件。
- 条件付きの注意点（§3）：m0の2件は物理的に同じroundだが、元fileが幅hのcell（S2・0.1と共有）に由来する。ユーザーがこれを流用に当たると判断した場合は、最大50件になる。

## 1. 方法：既存経路の再利用

1. **configurationの再生成**
   - T2の手順（`5fb2c32e scripts/run_h4_truncated_t2.py`の`plan_cells`）を使う。LD1のbase specに`D.full_pf_spec`を適用し、B0Iはprefix 7の分割を設定して`C.configure`に通す。
   - 6件とも、CのCELL_PLAN_V2のconfiguration IDと一致した。BのprovidersでもIDが一致した（B_INTERFACE_CHECK）。
2. **計画**：既存のC baseline経路（`I.plan`→`C.prepare`）を使う。
   - targetはC manifest（`c5466946`）の共有target。
   - referenceのdescriptorは保存pair（`bba15083`、`rounds[m].diagnostics.reference_pair`）。
3. **S2 meanの束縛**
   - 前回coverageでS2と判定したroundについて、現行SOURCEの保存direct meanを、baseline recordの形に包んで**無変更の`C.b_mean`**に通す。
   - provider bindingは、`I.produce_signals`と同じ式で付ける（`physical_context=I.price_context`）。
   - 対応表：

| field | 束縛した内容 |
|---|---|
| raw＝corrected | 保存`direct_PF_mean`（値は無変更） |
| B／logB／attenuation | 1／0／1 |
| law | deterministic（configurationのまま） |
| reference | bba15083の保存pair `reference`。`signal_difference`は既存の閾値1e−10以内であることを確認しただけ |
| numerical | 保存recordの数値gate（step・power unitarity、norm、閾値1e−9）をそのまま複製し、`gates_passed`を付ける |
| bias | null（staging中には算出しない） |
| scalar | 値の和を計算せず、規則と成分を束縛する。Full PFは「Hamiltonian定数を総Tで一回」で、定数は保存cellの`scalar_constant_ha`。B0Iは「定数＋p7 tail identityを総Tで一回」で、tail identityは保存contextへのlocatorとSHA（値は複製していない） |
| source | packetのsourceはstaging caller（`3d64d6891`）。元のsaved file・元cell・元round・producer SOURCEは`provider_binding.input_source`に記録 |
| canonical ID | 各roundの元canonical ID（`identity.candidate.id`）を保持。Bのtotal viewでの新IDは、人工fixtureでの確認値をB_INTERFACE_CHECKに記録（例：Full PF S2・16h/7は`af18d37d…`、B0I S2・16h/7は`e496358b…`） |

4. **S2LとS3**：`C.unavailable`のまま残す。`raw`・`normalization`はnullで、0や推定値を入れない。

## 2. Bの既存intakeとの接続（B `9c0e12d87`、無変更のcheckoutで確認）

| 確認 | 結果 |
|---|---|
| Bの回帰test（`check_rpe_total_baseline_binding` 7件、`check_rpe_fixed_grid_total` 6件） | PASS |
| BのprovidersでのconfigurationのID | 6件とも一致 |
| 実際の段階packet（公開commitのSOURCE） | `T.bind_signal`が**拒否**した。理由は「single-source signal provenance differs」で、未取得roundのsourceが異なるため |
| 未取得roundにpacketのsourceを付けた変種 | **拒否**された。provider bindingがないため。全roundの揃いを求める条件は緩めていない |
| 人工の完成fixture（取得済みroundは段階packetのまま、未取得roundは人工値とBの物理contextで埋めたもの） | **受理**された。取得済みroundのraw・reference・normalizationは変化しなかった。science callは0 |

**読み取れること**
- 段階packetの取得済みroundは、identity・law・物理context・provider bindingの点で、Bの既存checkを通る形になっている。
- 人工fixtureが受理されたことは形式上の確認であり、実H4の全roundを受け入れたことではない。

## 3. 由来に関する注意（S2 roundの元file）

- 早期roundのmeanは、A H_eff scanの別幅cellやT2の幅cellが算出したものである。physical IDは同じ物理roundを示す。
  - 例：Full PF Y4 m0はscanの4th / 0.1 cell、M5 m3はscanの4th(m5_best) / 0.8 cell。
  - これらの対応は、A follow-upの計画（`C_HANDOFF`の`effective_spectrum_source`）が既に指定していたものである。
- **Full PF S2・16h/7のm0**の元fileは、S2・幅0.1のH_eff pilot（`69ee0ff8` round_0）である。
- **B0I S2・16h/7のm0**の元fileは、T2の幅h cell（`cell_p7_2nd_1`。S2・0.1とも共有するcell）である。
- いずれもS2・0.1の13round packetを流用したものではなく、物理的に同じround（q1、T=0.1、幅0.1）である。
- ただし指示の「S2・0.1の保存信号をshortlistへ流用しない」に当たるとユーザーが判断する場合は、この2件をS3へ移す（各1件増）。

## 4. S2Lの技術調査（[S2L_AUDIT.json](S2L_AUDIT.json)）

対象は、Full PF S2・Y4、B0I S2・Y4の各m3（T=0.8）。いずれもTrack Aの旧meanである。

| 確認項目 | 結果 |
|---|---|
| 元のH・state・snapshot | 現行と一致（pf_commonの`source_input_identity`） |
| physical ID | 一致 |
| 元の数値SOURCE | Track A pf_commonの`e2179f68`。長時間study（`58b21d596`）のrecordは、pf_commonの値を再利用している（独立ではない） |
| 算出経路 | pf_commonの`h4_signals`（provenance pointer `/signals/N`）。保存fieldはraw・corrected・reference・log_B・total_signed_bias。gate証拠は`allowance_evidence`（maximum_intermediate_norm、raw_corrected_vector_closure）と`old_u`・`old_margin` |
| mandatory STOP | Track A studyは完了済みでMANDATORY_STOP |

**現行の受入に足りない証拠**
1. 現行policyの数値gate：step・power unitarityとnorm errorの値（閾値1e−9）。
2. 現行のdirect経路（`DIRECT_EXPM_SIGNED_PF_MATRIX_POWER`）と同じbackendであることの証拠。
3. このround objectについて、bba15083のreference pairに対するgate証拠。Track Aは独自に保存したreferenceを使っている。

**判定**：**既存記録だけでは受入条件を確認できない**（`PENDING_NOT_PROMOTED`）。
- 前例：C `c5466946`は、S2・0.1のTrack A m3/m4を、power診断が欠けているとして新規directで評価する扱いにした。
- 参考（資格ではない）：同じTrack A経路のM5 m3/m4は、現行SOURCEの値と4.8e−13以内で一致している（前回coverage）。

**次の科学計算候補**（未承認）：保存stepに対して、このroundのdirect powerとgateを1回ずつ計算する（4件）。新しいmeanの算出と実質的に同じ費用になる。

**screen run02のm6について**：Full PF S2・0.1だけに関係し、6候補の対象外である。区別したまま、扱いは変えていない。

## 5. S3：最小の追加計算一覧（承認待ち、今回は未実行）

| 候補 | round | 件数 |
|---|---|---:|
| Full PF S2・16h/7 | m2、m5〜m11 | 8 |
| Full PF Y4・16h/3 | m5〜m11 | 7 |
| Full PF M5・16h | m5〜m11 | 7 |
| B0I p7 S2・16h/7 | m2、m5〜m11 | 8 |
| B0I p7 Y4・16h/3 | m5〜m11 | 7 |
| B0I p7 M5・16h | m5〜m11 | 7 |
| **合計** | | **44** |

- 条件付き：S2L（m3）を使わない場合は＋4。§3のm0を流用とみなす場合は＋2。
- 各stepの固有系・演算子は保存済み（Full PFは`69ee0ff8`・`3ae055fa`・`a3005d72`、B0IはT2）。
- 計算するときは、既存のC baseline `execute`（`saved=`経路）と既存のgateで、全13roundを一つのpacketにまとめる（HANDOFF_B §2）。

## 6. ロードマップへの影響（§10.6。本体は編集していない）

| 箇所 | 更新すべき内容 |
|---|---|
| §4.2・§5（Full PF・Truncated shortlist） | 「全13round packet未同定」から、「保存meanのSOURCE付き受入準備が30 round完了（6候補で4〜6 roundずつ）。S2Lが4、S3が44」へ更新する。packet化は済んだが未完成で、risk・N入力ではない |
| §7（Full PF全13round信号） | 再利用できる分のpacket化が完了した。残るのは、44（＋4、＋2）件の新しいmeanの算出承認と、S2Lの扱いの判断 |
| 依存関係 | 決定論PFのG構成は、A（44件の計算）→Aの全13round packet→Bのintake（既存経路のまま、変更不要）→B risk・N、の順に依存する |
| 承認 | 新しいmeanの算出と、S2Lのpower・gate計算には、科学実行の承認が必要 |

## 7. 保全とSTOP

- 元のSOURCE・raw・保存mean、A/B/Cの固定成果、grant・ledger・marker・STOPは変更していない。
- Bのcodeは、自分で作ったdetachedのcheckout（`9c0e12d87`）で実行しただけで、Bのbranchには触れていない。
- B・C・Dのworktree・process、共有環境は変更していない。
- **mandatory STOP**：不足meanの新規計算、実物理条件での再資格化、MSE・N・G、compileへは進まない。
