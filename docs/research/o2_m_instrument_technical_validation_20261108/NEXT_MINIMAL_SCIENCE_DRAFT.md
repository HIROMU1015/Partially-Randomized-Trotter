# 次の最小科学検証草案：H4の1 slotに対するmetadata-only適合確認

**提案のみ、未採択・未実行。** 新しい実state/Born/shot/compile・variance/N/期待費用評価を含めない。今回の人工PASSを理由に旧科学契約を変更しない。

## 固定する入力とSOURCE案

- 前段設計：`154c6e3cf1f48d27f558003b0e9bd74882414d9e`。今回の新コードは本草案と同じ公開commitへ固定し、[run03 SOURCE](../../../artifacts/o2_m_instrument/2026-11-08/artificial_run03/SOURCE.json)のcontent hashと結合する。
- N1結果：`744e1a09661ebba6841fe5faf456f88c41c47780`、N1科学SOURCE：`b6dbec9f80da35a88fba28795ad02ca74c9adfa0`。
- 読取候補：N1の`artifacts/o2_a1_new_target/2026-11-06/n1/`に保存された`SOURCE.json`, `CONFIG.json`, `LAW.json`, `POLICY.json`, `PREPARATION.json`。Git objectの固定blobと保存identityを先に照合する。
- 条件：保存linear H4／1.00 Å／STO-3G／legacy DF12、L_D=3、m0、q1、δ=0.02、r1、K2、coefficient_atol=0.0、8 system qubits、N1 actual finite-grid Qとselected full policy。Hamiltonian/partition/preparation/tail/law/policy identityを変えない。
- N2結果`5e9bd08068738cdf1070bdc43bee8194b113ae33`と旧engine認証`fdf00cee3c87d804845497afc89ac6a71f3b3f59`は後段での状態/targetの参照先。**このmetadata検証ではstate・spectral witnessを入力にせず、旧engine/ground loaderを呼ばない。**

## Slotの事前選択と有限scope案

**m0/q1/r1のn=0 eventで、唯一のrotation application slot**を選ぶ。価格や観測結果によるslot選択は行わない。n=2等の別contextや別roundへ広げず、元targetのそれらの寄与を除外・変更もしない。今回確認するのは条件付き部分への適用可能性だけである。

N1に保存された当該slotの有限label catalogを一回読み取り、各labelを一回ずつ記述化する。全path/label組合せ・state・行列の列挙は不要。全zero-grid labelは元lawとともに記録し、群候補は正質量labelに限定する。保存catalogを越える候補探索、basis再選択、matrix fitting、数値対角化、compileは行わない。

共通A/Bは既存`df_partial_s2`と`SavedEngine.selected_circuit`の**静的な構成順序**から確認し、Vは実際のselected定義を使う。後者を実入力で呼ぶとfull回路/stateへ接続するので呼ばない。original basisの一致や保存residualの小ささだけでselected Vを共通化しない。

## 最小binding差分と予定するcall

現在の人工adapterはQiskit circuitからgate署名を得る。metadata-only段階には、保存されたoperation記述を**量子回路をbuildせずに**同じ意味のliteral gate tupleへ写す薄いheader binderが追加で必要である。このbinderは今回未実装。現在の人工builderをH4へそのまま実行できるという契約にはしない。

予定callは、固定rootとfresh outputを受け取る**一回のmetadata-only pass**（仮称`inspect_saved_n0_slot_headers`）。SOURCE/環境・header codecを実科学入力投入前に固定/公開し、その実装、入力、call=1、retry=0、出力を別途明示して承認を受ける。保存数値の新集計に当たるため、今回の技術許可では開始しない。

pass内では、以下だけを得る。

1. selected operation・phase・wire・A/B/V/φのliteral記述と元targetへの対応。新しい行列積・unitary検算は行わない。
2. 同じη/slot/policy/lawと共通署名に基づく正labelの分割。未知/未対応は保留せず明示分類し、正supportを消さない。
3. p_g、q_l=p_l/p_gを保存actual整数質量から正確に取得。新CDF/preparationや理想係数の再正規化を使わない。
4. 少なくとも2つの正labelと異なるsigned diagonal functionを持つ群の有無。これは非自明なmetadata群の条件であり、実stateでの|r(z)|<1の正確率や分散利益ではない。
5. A=B†に相当する明示構造の有無を分かる範囲で記述し、強いancilla-free直接測定を比較対象から落とさない。一般operator等価性の探索はしない。判別できなければUNKNOWN。

全system outcome zのr/m値生成、最良群の価格最適化、grouped state evolution/Born、shot/RNG、compiled価格・N・総費用はこのpassに含めない。機械可読出力はinput identities、slot記述、各labelの適格/不適格理由、群と質量/q、target対応、UNKNOWN、call記録、STOPに限定する。

## 判定とSTOP

入力identity/codec/phase/order不一致、selected targetを表現できない、未対応operation、正support欠落、actual mass不整合、出力衝突でSTOP。近似一致・新basis・別slotで救済しない。

非自明群がなければ「当該slot/選択済み記述で非自明群を取得できず」と報告し、一般的なMのNO-GOにしない。群があっても「metadata適合」でSTOPし、Born/shot/native価格へ自動進行しない。共通VやA/Bのoperator等価性しか分からずliteral適合が得られない場合はUNKNOWNのまま残す。

次のユーザー判断は、この1 slot・一回のmetadata-only passと必要な薄いheader binderを許可するかである。測定方式の正式採択、実H4の新instrument誤差認証、有限shot検証、full資源比較は、それぞれ後の実行範囲・SOURCE・STOPを固定して別途判断する。
