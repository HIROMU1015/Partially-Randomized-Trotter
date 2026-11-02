# A2 認証情報接続設計：ユーザー／GPT向けハンドオフ

2026-10-18 JST。**技術分類：`DESIGN_READY_FOR_IMPLEMENTATION_PLANNING`。** 後続Codexが何を実装し、どの証拠を再検証するかを定めた。実装済み・certificate取得済み・実行準備完了・A2採択という意味ではない。

成果物は[接続仕様](certificate_connection_spec.md)、[既存APIと実装作業表](source_api_connection.md)、[暫定比較・費用台帳](comparison_and_cost_design.md)、[証拠索引](evidence_index.json)、[公開receipt](publication_receipt.json)。

## 1. 前回から閉じた仕様

| 前回の不足 | 今回具体化した内容 |
|---|---|
| 同じモデルを参照する仕組み | rawとexact係数を別hash化し、model→sector→partition→PF→bundle→comparisonの親参照を固定。full/correction、scalar、mode順、targetを区別 |
| 丸め・Hermiticity | binary64のexact decode、exact Hermiticityの受理/拒否、silent Hermitization禁止。exact Ratのcoreと表示用enclosureを分離 |
| PSD/L/f/Λの取得手順 | K0の符号別外側shiftを、既知の係数三角capで必ず検証できる設計へ接続。L・負平方scalar・shift後capを同じ帳簿で定義 |
| Uの根拠 | 明示occupation determinantの正規化/N/CAR遷移をexactに検証する変分上界。HF metadata/FCI/近似固有値を証拠にしない |
| K_SOS | K0 raw辞書・添字/重複・sqrt上界witness・行和を生成側と検証側へ分離。実装のbit/保存費も記録 |
| 強い対照 | center capsと、grouped double commutatorのexact CAR/JW係数・相殺を使う経路を許す。同じ情報と取得価格を公開し、粗い対照だけで勝たせない |
| stepの出口 | RF/EW2とcentered Cbar2の各domain・最大十分stepのsup・検証可能なbracket・未解像/退化case・比較不能を定義 |
| access/費用 | exact ground copies条件付き比較と、準備unitary・近似state・実QPEを別scopeへ分離。生成/検証/償却、state/powers/native/readoutの未知費をnullで保持 |

これらは既知のPSD shift・CAR/変分・norm評価を認証formatへ接続する技術設計であり、新しい数学的保証の研究ではない。粗いcapの有用性や構造展開の実費はまだ不明である。証拠型の定義はcheckerのcorrectnessの証明・platform certificateでもない。

## 2. 暫定科学前提と未採択事項

ユーザー指示で今回の設計前提として指定されたものは、固定DFモデル、初期は固定N、実装可能なPSD分割の正stage S2、exact ground accessの条件付き入力、同じ情報によるRF0/RF1/EW2と強い交換子対照である。これを正式研究採択・科学実行契約へ読み替えていない。

後続でユーザーが採択する必要があるのは、具体的なraw/model・mode ordering・N・block順序/PSD cap policy、U用trial determinant、d・α_specと実QPEへ進む場合の精度/失敗率配分、root/sqrt enclosure幅・算術/構造評価policy、比較範囲と費用償却policy、科学call/保存/STOP/retry条件である。今回、分子・geometry・rank・p・threshold・call数・慣例confidenceを埋めていない。

固定DFのtargetは、元電子Hamiltonianからの変換を認証したことではない。元分子への拡張、近似stateの保証、state preparation/natives込みend-to-end評価は別の科学的選択と証拠を要する。追加GPT科学判断が必要なら、ユーザーが範囲を指定して開始を承認する。今回新しいレビューを開催していない。

## 3. 次の技術作業と承認

**次の担当はユーザー。** この設計を基に、数値実行を伴わない限定SOURCE実装・検査仕様の準備をCodexへ別途指示するかを判断する。今回の依頼は実装の承認ではない。

別途指示を受けた場合のCodexの作業単位はSOURCE表のSPEC-01–07（exact intake/ID、sector/PF/scalar、L/f/Λ、U/W、K_SOS、Cbar2、step/comparison checker）である。既存実装をそのまま認証済みとして接続できるわけではなく、これらのadapter・generator・独立verifierは未実装。数値人工テストも今回実施していないので、後続の実装許可に検査範囲を明示する必要がある。

その後、ユーザーが具体条件・SOURCE/入力identityを正式契約として採択し、**さらに別途、certificate生成・保存入力検証・step比較の科学実行を明示承認**する順序となる。将来のcertificate再検証も科学的演算を含み得るため、静的検査やdry-runという名で実入力を評価しない。native/QPE/状態準備の実装・実行へ自動移行しない。

実行前に必要なのは、入力raw/model規約の確定とexact Hermiticity/N文法、PSD scalar台帳、L/U/f/Λ、同順序K_SOS/Cbar2、generator/checkerのSOURCE・検査・算術仕様、共通d/α・境界解像度と保存条件である。入力やcertificateは本designには存在しない。

## 4. 残る限界と分類の理由

固定DFのidentityを定義する仕様、粗くても閉じるPSD/cap/Uの既知方法、K0辞書/strong Cbar2の検証手順、step出口と比較不能/費用scopeまで定まったため、実装計画を立てるための技術設計は完成した。

実入力やtaskの未採択値は**将来の科学契約の未決事項**であり、仕様の型や証拠methodが不明という意味ではない。このため`DESIGN_READY_FOR_IMPLEMENTATION_PLANNING`と分類する。実際のrawがHermiticityを満たすか、粗いcapsで有用なstepが出るか、exact algebraが安価か、十分なstate accessがあるかは未検証。前回の実例準備に関する`INPUT_CLASS_FOUND_CERTIFICATES_INCOMPLETE`を「解決済みcertificateあり」へ上書きしない。

特に、比較結果が出ても対象は宣言したcentered Cbar2対照との十分証拠上のstep差まで。低energy/回路安定性を含むbest-known envelope、真のPF bias、元分子の保証、状態準備込み総費用、研究の新規性・GO/NO-GOは認定しない。

## 5. 来歴・保全・公開

入力は[ユーザー指示のbyte保存copy](inputs/user_instruction_a2_certificate_connection_design_20261018.txt)。必須preflight本文・SOURCE表・handoff・index・receiptを取得し、その固定commitを基点にした。数学・正式状態・最新共通運用指示の固定refと、再利用したSOURCE inventoryはevidence indexからたどれる。必須指定資料の未取得はない。SOURCE機能の再調査・新しい証明レビュー・他repositoryの再監査は行っていない。

専用branch `qc-a2-certificate-connection-design-20261018`、独立作業領域 `/tmp/prt-qc-a2-certificate-connection-design-20261018`。新しい専用ディレクトリのみ追加し、旧SOURCE・raw・契約・レビュー原本・authorization・grant・ledger・markerを保全する。親worktreeのHEAD/local branches/registry/index/exclude/tracked bytes・選択原本と、独立作業領域の全base blobを開始状態と照合する。研究概要/PROJECT_MAP/validation inventoryは変更しない。

正式状態は[固定decision `67576526…`](https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/67576526c4e96ae7235fa645fdf744278d049f8d/docs/research/qc_exact_encoding_direction_adoption_20261017/decision.json)のまま。A1 CLOSED、A2 NOT_ADOPTED、K1-F未承認・未実行、Exact Encodingの現構成REDUCES_TO_KNOWN_METHOD、他CLOSED/STOP/PAUSED/HOLD、①②③の状態も変えない。

共通運用指示2026-10-17 §8.4に従い、人工的な低資源上限・共有環境や他jobへの変更を導入していない。今回の実作業は文書読取・設計とGit/JSON/link/hashの管理検査だけである。新規科学計算、保存数値再解析、certificate生成、新数学保証、科学SOURCE/runner、科学実行契約、authorization/grant/marker、新重要科学レビュー、A2採択は**各0件**。

本文payloadをcommit・pushして公開bytes/SHA256を照合後、publication receiptを子commitへ保存する。receipt自身のhash/最終commitの自己参照は避け、最後に最終remote HEADと全公開ファイルを再確認してチャットで報告する。**公開・保全確認後mandatory STOP。**
