# Track B E/T：限定計算と保存値集計の完了／GPT引き継ぎ

**`SAVED_E_T_ACCOUNTING_COMPLETE`**。元v7のnative22、登録T key39の計算が完了し、
保存不具合を修復して同じsourceの会計を保存値から復元した。復元の外側exitは0。
**mandatory STOP**している。研究方針・新規性・算法採択・総PR資源の勝敗は判定しない。

重要な区別：**元v7 one-shotは`TECHNICAL_INCONCLUSIVE`のまま**である。
native/Tを完了後、集計JSONの保存で停止した原terminal・marker・結果を変更していない。
今回の完了は、そのrunの既存結果から会計を復元した別のsaved-value recoveryであり、
元の科学runを正常完了と再分類したものではない。

## 固定identity

| 項目 | 完全SHA／値 |
| --- | --- |
| Repository | HIROMU1015/Partially-Randomized-Trotter |
| completed recovery branch | `track-b-df-correction-et-saved-recovery-v2-20261012` |
| science source S7 | `74682c40d0cdcca92d244a9edf4bb2852361eef5` |
| science execution authorization A7 | `84562f8c5af9249a902662324200456498a83d15` |
| science result commit R7 | `ad4b4a334171cc39053c0804876a63c93093bfa3` |
| science marker SHA256 | `28e8d00ab4d8703e3693c29d946011c27c17cab2ddb3650d783ddeb8197b9dc2` |
| recovery source S9 | `3afef9e418a8d2173a2780f5534404390bfbe6fb` |
| recovery execution authorization A9 | `eea9c497421c5dcfc541dcaa508f607112dbd0d5` |
| recovery marker SHA256 | `452325155c8f5166398753b36ac852863b11e3a976210ffabb1a41bd3f25bf7d` |
| original post-T AST SHA256 | `7246162e347f77b7a14552955095c7fc3b0a0884a015926ed69607c2aae3d4de` |
| recovered logical accounting bytes | 1,135,179 |
| recovered logical accounting SHA256 | `b9a6d6aa684f4df4808c1b59d33782cd83f36afad1d510dec5c89a758b0f6ad4` |

結果commitとGitHub再取得検証は新auditの`publication_receipt.json`に記録する。
receiptを追加した最終公開HEADは完了報告で提示する。
公開後HEADと上表のexecution HEADは区別する。
原科学dataは`artifacts/track_b_df_correction_et_resumption_result/2026-10-12/v4/`。
復元dataは`artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/`。
新auditは`artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/`。

## 利用者認可と修正履歴

利用者の「次も停止した場合についてもそのまま修正作業に入って、計算完了まで進めて」と
中断後の「続きを行って」に従い、技術修正・新source・新authorizationへ分けて継続した。
old markerを再利用せず、各runはretry0。以下の複数実行をcampaign全体一回とは表現しない。

1. **v6**：guard式・threshold不変で失敗前のupper／ledger／IRを保存。
   一回実行してr6 controlled Aの停止を確認。native10、T0。
   nearest Wへの未実施置換距離約0.098が支配項で、他の四項は合計約4.41e-8だった。
2. **v7**：raw phaseを保持したまま、非対応Wへの実施しない置換距離を加算する不整合を修正。
   strict full Operator、reference／forward／angle guard、2e-7、比較条件・capsは保持。
   supported Wの距離は計上し、unsupported phaseのT費用はunknownを維持。
   一回実行でnative22とT key39を完了したが、large unknown dictionaryのJSON保存で停止した。
3. **saved recovery v1**：大きい`unknown_multiplicities`辞書をblobへ分割。
   新result親directory作成漏れでmarker作成前に停止。追加science0。記録を保存した。
4. **saved recovery v2**：起動修正とregressionを追加。
   同じv7のpost-T ASTを保存値だけで処理し、正常完了した。追加native／T／matrix／LP／sampling0。

latest technical continuation内のE/T実行はv6/v7の2回、native合計32、T key39。
recovery起動は失敗／成功各1回。各epochの原結果・失敗・source・authorizationを保存する。
旧v4/v5と以前の停止も不変である。

## 比較条件・会計のscope

Eは保存済みsynthetic r3/r6、rank2×3、seed7101107の元生成順、sign(+1,-1,+1)、
theta1/32、kappa4、u1/128、uncontrolled／MSB-controlled。
active-frame identityの局所centre費用であり、分子geometryやproduction DFではない。
Qiskit1.3.0、rz/sx/x/cx、opt0、seed20261011、numeric2e-7／synthesis8e-7／full1e-6を維持した。
native最大22、T最大64、各2workers／threads1。pygridsynth2.0.0、dps100、up_to_phase=false、
gate precision1/40000000000、元key選択・seedを維持した。

CD6taskは旧runの保存substage参照だけ。新しいsignal・shot・Hamiltonian・NPZ・trajectoryは計算していない。
CDの補正頻度を別のE fixtureの単価へ掛けて実測PR総費用にしない。
sourceの`CD_PR_total_winner=null`、未知費用を0にしない方針を保持する。

## 保存された4条件のT会計

この表のA lowerは**取得済みsequenceを用いた固定additive planのcompletion lower**である。
最適compiler／全algorithmの下界ではない。B upperは同じlocal correctionに対する既定planのworst-event T。
両方ともsampling moment、outer coin、全PR costを含まない。

| E modes | controlled | A fixed-plan T lower | A actual total T | B complete-plan T upper | B expected conditional T | A unknown RZ occurrences |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| 3 | false | 576 | unknown | 112 | 112 | 57 |
| 3 | true | 692 | unknown | 236 | 236 | 179 |
| 6 | false | 576 | unknown | 116 | 116 | 5,519 |
| 6 | true | unknown | unknown | 232 | 232 | 11,622 |

sourceが出力する`FIXED_PLAN_LOCAL_T_SEPARATION`は最初の3行でtrue、最後の行でfalse。
最後の行はAのunsupported scalar phaseにより既定lowerがNoneであり、falseを非優位の証拠とはしない。
全Aのactual totalは未確定。全B eventの対象keysは取得済みで、規定のerror allowancesを満たす。
raw phase・nearest-W distance・未確定scalar phase実装費用をnative inventoryへ残している。
全PR資源、native Gaussian embedding、分子・DF規模での性能、新規性の結論はこの表から出さない。

## provenance・復元と検証

原v7 resultは630ファイル、39,600,264 bytes。native22、unique T key39（backend33、Clifford fast path6）。
全keyのsequence SHA256、T/T†/W count、strict error guardとinventory/native bindingを保存した。
元one-shot外側wall326.7044s、CPU586.1332s、peak sampled aggregate RSS732.7305 MiB、exit2。
元理由は`ValueError: Review file exceeds 512 KiB`。資源上限超過は記録されていない。

復元は元sourceの`sequences={}`から`T_accounting`保存までのASTをそのまま使った。
保存値の署名・入力・sourceを照合し、元jobのabsolute output path stringを保持した。
新しいmatrix／circuit／synthesis／candidate選択は行わない。
原accounting JSONは保存されなかったため、失われた元bytes/hashとの直接比較はできない。
根拠は同一AST、全保存入力identity、独立したcount再集計と元formatterのlossless復元である。
大きい未知key辞書を256 KiB以下UTF-8 chunkへ分離し、全新fileは512 KiB以下。

復元結果は18ファイル、942,681 bytes。
recovery-only外側wall11.9180s、CPU11.4433s、peak sampled RSS251.8555 MiB、exit0。
原contractのCPU/RSS/wall/output/log上限を維持し、外側Supervisor／kernel CPU guardを使用した。

**最終saved-output監査：815 PASS、0 FAIL。**
原630ファイル前後一致、native IR/count/phase/accepted guard、全39sequence identity／error、
per-key CPU/RSS、4行のknown/unknown T／B期待値、marker・terminal・outer statusを照合した。
原runは再分類せず、復元で新science0を確認した。
new source固定前のlocal focused testsはv6=134、v7=91、recovery=30、startup修正=31 PASS。
全失敗test、deselection理由、起動失敗は保持する。immutable CI／外部再現とは呼ばない。

## GitHubから取得する入口

- [全review evidence manifest](../../../artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/evidence_manifest.json)：原630ファイル、復元18ファイルと必要source・監査・文書のpath／bytes／SHA256。
- [saved-output監査](../../../artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/saved_output_audit.json)：815 PASS。
- [最終provenance監査](../../../artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/final_provenance_audit.json)：797 PASS、旧4worktreeのHEAD／cleanと凍結sourceを照合。
- [4条件のexact保存値](../../../artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/comparison_rows.json)。
- [復元accounting descriptor](../../../artifacts/track_b_df_correction_et_saved_recovery_result/2026-10-12/v2/T_accounting.json)：descriptorが参照するtyped snapshot／immutable blobsの全chunkもmanifestに収録。
- [STOP](../../../artifacts/track_b_df_correction_et_saved_recovery_audit/2026-10-12/v2/STOP.json)。

各ファイルは512 KiB以下。分割は同じlossless codecのlogical JSONへ戻す形式であり、表示用の近似値で元exact値を置換していない。

## GPTへ戻す判断と停止

今回の完了は、事前登録された有限scopeの計算・会計・取得資料が揃ったことを意味する。
A/Bのsampling/bias/resource tradeoff、強い既知法との関係、whole方式を主線として維持する価値を
GPT側の科学的レビューで判断する。レビュー開始は利用者側の判断に従う。

必要資料をcommit/pushし、remote SHA、GitHub再取得、bytes/hash、cleanを確認して引き継ぐ。
**mandatory STOP。追加source修正・science・T合成・fixture/grid/precision追加、G10／RA-D0／分子／GPUへ進まない。**
