# G6適用可能性の多軸一覧表

2026-10-18 JST。本調査の最重要成果物。評価対象は、各演算子を**既存のG6入力・生成結果として現行PR/RTEへ実際に接続できるか**である。primitiveの存在だけを直接接続と判定しない。

## 判定語と優先順位

`SUPPORTED`＝当該列の条件が固定証拠で支持される。`IMPLEMENTED_PRIMITIVE`＝既存SOURCEにprimitiveの構成がある（今回の実行・native認証を意味しない）。`NOT_ESTABLISHED`＝資料未取得・実装未確認・未検証。`NOT_APPLICABLE`＝明記した演算子・空間における条件不適合。

総合分類の略号は次の完全名に対応する。

- **DIRECT**：`DIRECTLY_SUPPORTED_BY_EXISTING_SOURCE`。今回のG6実接続条件まで既存証拠で確認できる場合だけ。
- **INCOMPLETE**：`ALGEBRAICALLY_VALID_IMPLEMENTATION_INCOMPLETE`。
- **NEW_CHOICE**：`REQUIRES_NEW_DECOMPOSITION_OR_RESEARCH_CHOICE`。G6適用に新D/R・方針選択が要る場合、実装未完了より優先。
- **UNSUPPORTED**：`NOT_SUPPORTED_OR_NOT_APPLICABLE`。補助statusによって資料不足と不適合を分離。

以下の「古典取得」は情報と経路の確認であり、CPU時間の実測ではない。「native費用」は必要項目の特定であり、回路価格の新計算ではない。

## 候補ごとの全軸

| 候補・演算子 | 代数：Hermitian／involution | 演算子・係数の取得／oracle | 既存SOURCE対応 | controlled実装 | 現行PR/RTEのR接続 | 古典取得 | native費用 | 既知対照との比較可能性 | 資源改善の実証 | 総合分類・補助status |
|---|---|---|---|---|---|---|---|---|---|---|
| **C01** `s B_l Z_p B_l†` | 両方SUPPORTED | c、sign、basis ID/hash、U_opsの経路SUPPORTED。調査用の分子固有係数bundleは未取得 | S09/S12/S13。既存Rの抽出成分 | **IMPLEMENTED_PRIMITIVE**：無制御basis＋CZ／controlled-RZ＋ancilla sign位相 | Rに同じ成分がある：SUPPORTED。**G6 packet→event→measurement：NOT_ESTABLISHED**。新D/R不要 | full-basis取得経路あり。rational入力binding・angle取得費用は未確認 | basis開閉、CZ、controlled-RZ、phase、誤差認証が必要。G6価格NOT_ESTABLISHED | ordinary DF-native、Pauli展開/CTS、Gaussian subcircuitを同条件で比較可能。実比較なし | **NOT_ESTABLISHED** | **INCOMPLETE**。一般には非Pauliとなるクラス。特定分子全成分の非Pauli認証は未実施 |
| **C02** `s B_l Z_pZ_q B_l†`, p≠q | 同一basis内の可換性により両方SUPPORTED | C01に加えて2軌道supportとc_pqの式 | S09/S10。既存Rのdensity-density展開成分 | **IMPLEMENTED_PRIMITIVE**：2 CZ／controlled-RZZ。回転は一般にGaussianでない | C01と同じ。**G6実接続NOT_ESTABLISHED**、新D/R不要 | full-basisあり。thin orbital列・G6の係数/角度契約は別の不足 | basis＋2 CZ、controlled-RZZ、phase。G6価格NOT_ESTABLISHED | C01と同様。Gaussian専用圧縮の適用範囲をterminal ZZ回転へ無断拡張しない | **NOT_ESTABLISHED** | **INCOMPLETE**。同一basis可換parityと異基底積を区別 |
| **C03** `n_lp`そのもの | Hermitian/projector：SUPPORTED。involution：**NOT_APPLICABLE（full Fock）** | rotated numberの意味はL01/L02/S13に対応。係数は元項に依存 | S13/S24。G6のQ型ではない | number evolutionの既存構造はあるが、これをG6のQとして制御しても条件不適合は解消しない | そのままのQとして**NOT_APPLICABLE**。`(I−T)/2`への既知変換はC01とshiftへ戻す | 変換後の符号・定数・係数を残す必要 | 変換後のcentral rotationとancilla scalarが必要。新価格なし | projector-to-involutionの同じ恒等式を全対照に認める | 本候補として**NOT_APPLICABLE**。変換後の改善はNOT_ESTABLISHED | **UNSUPPORTED**／`NOT_APPLICABLE_AS_WRITTEN_FULL_FOCK`。未取得という意味ではない |
| **C04** `n_lp n_lq`, p≠q | Hermitian/projector：SUPPORTED。involution：**NOT_APPLICABLE（full Fock）** | 既存I/Z/ZZ係数式あり | S09/S24、L02。単独Qではなく4項の既知対応 | 展開後C01/C02のprimitiveあり | そのままのQとして**NOT_APPLICABLE**。現行展開でC01/C02へ対応 | 展開・shift・重複p=qの処理を保存 | 一つのinvolution回転の価格に置換しない。新価格なし | 同じ展開とbasis共有をordinary/Pauliへ認める | 本候補として**NOT_APPLICABLE**。展開後の改善はNOT_ESTABLISHED | **UNSUPPORTED**／`NOT_APPLICABLE_AS_WRITTEN_FULL_FOCK` |
| **C05** `B_l λ_l(Ση_lp n_p)^2 B_l†`全体 | Hermitian：SUPPORTED。unshifted全blockのinvolution：**NOT_APPLICABLE（真空0）** | eta、lambda、U_opsの既存型あり | S14/S09。現行はI/Z/ZZの線形結合へ展開 | DF block evolution構成あり。単一G6-Q oracleではない | 全blockをQにする読み方は**NOT_APPLICABLE**。既存展開成分はC01/C02 | 分解済みinputと係数経路あり。新しいshift・分解は考案しない | full DF block価格と単一Q価格を同一視しない | 既知DF-nativeとの構造比較可能。全block一個のinvolutionという条件は共有できない | 本候補として**NOT_APPLICABLE**。別分解の利益はNOT_ESTABLISHED | **UNSUPPORTED**／`NOT_APPLICABLE_UNSHIFTED_FULL_FOCK`。固定粒子sectorについての一般no-goではない |
| **C06** 異基底`Q_a Q_b`を一つのgeneratorにする読み方 | unitaryはSUPPORTED。Hermitian/involutionは可換なら成立。**個別pairはNOT_ESTABLISHED** | 元の二成分accessはあり得るが、可換性certificateは未取得 | S02/S07。product roleとしての積とgeneratorを区別 | 順序付きcontrolled productはprimitive列で表せる。`exp(-iφ Q_aQ_b)`への単一involution loweringは未成立 | 無条件の単一Q扱いは支持されない。実際の個別pairを不可能とは断定しない | 可換性・係数・identityの照合が不足 | Gaussian積圧縮の旧候補を再開しない。価格NOT_ESTABLISHED | 既知Gaussian product法のscopeは比較可能。単一generator条件は未確認 | **NOT_ESTABLISHED** | **UNSUPPORTED**／`NOT_ESTABLISHED_PAIR_COMMUTATIVITY`。無条件の一般則は不成立、個別例の不適用確定とは別 |
| **C07** Dのone-body／prefix内の同形reflection | 両方SUPPORTED（C01/C02の条件） | D用のeps／eta、basis操作はS12/S13 | Dに対応。Rの抽出対象には含まれない | conjugated Z/ZZ primitiveの構造はSUPPORTED | **現在はD。Rへ移すには新D/R・研究判断が必要**。自動移送しない | 現行Dの入力はある。新しいR入力は未採択 | 将来の新task価格はNOT_ESTABLISHED | 原D/Rを固定した比較と、新分割比較を混ぜない | **NOT_ESTABLISHED** | **NEW_CHOICE**。実装未完了より新研究選択を優先 |

根拠リンクと式は[候補棚卸し](candidate_operator_inventory.md)、[PR/RTE接続](pr_rte_connection_and_access.md)、[controlled条件](controlled_native_feasibility.md)、[一次文献・対照](pauli_and_known_method_baselines.md)。機械可読な候補ごとの独立statusは[evidence_index.json](evidence_index.json)。

## この表が支持する結論

既存RのC01/C02について、応用演算子の実在性・係数の式・controlled primitiveのSOURCEは具体的に確認できる。G6全接続と精度・資源の証拠が残るため、**DIRECTに分類する候補はない**。これはG6接続が不可能、資源改善が不可能、または投稿不適格という判定ではない。

非Pauliというクラスの性質だけから情報取得優位や性能優位を認定しない。元PR論文もこのクラスを明示している。C03–C05の不適合は「そのままの単一Q」「full Fock」という条件であり、既存のI/Z/ZZ展開の適法性を否定しない。C06は資料不足・条件未確認を保った。科学的GO/STOP、研究方針・独立新規性は本表で決めない。
