---
name: rpe-cost-compare
description: "Partially-Randomized-Trotter専用。Hamiltonianと要求エネルギー精度から、B2の共通プログラムでRPE方式別の最安適合構成・shot配分・MSE・compiled RZ費用を比較する。保存結果の再利用と承認済み予算内の実行に用いる。"
---

# RPE方式別コスト比較

利用者のHamiltonianと要求精度を、B2の**共通プログラム**へ渡して比較する。
候補生成・shot配分探索・価格計算・Q3・候補選択・予算消費・再開はそのプログラムに任せる。
Skill内に別optimizer、科学caller、価格モデル、予算台帳を実装しない。

## 最初に確認するもの

リポジトリrootで作業し、適用されるAGENTS.md、採択済み研究条件、
利用者の今回の許可範囲を確認する。実行入口と接続確認の状態は
[references/program-entry.md](references/program-entry.md)を読む。

**未公開・未接続の入口を推測で実行しない。** 公開プログラムがまだ無い場合は、
利用者入力と未接続箇所を整理し「Skill作成済み・接続未確認」と報告する。
過去の個別Q3 runnerや別の保存catalogを代替の共通optimizerとして起動しない。

## 入力を一括して受け取る

受け取るのは、Hamiltonianの識別子またはデータSOURCE、要求エネルギー精度と単位、
必要なら方式範囲・研究条件・有限実行予算・保存結果のみか新規実行を含むか、である。
同じ会話で既に指定された条件は引き継ぐ。

- 「H4 DF10」等の別名は、公開共通プログラムの問題設定・入力receiptに照合して解決する。
  geometry、basis、sector、H/state/snapshot、energy targetを別候補と混ぜない。
- 精度は十進表記と単位を保持し、RMSEとMSEを区別して正式なprogram入力へ渡す。
  採択済みdecoder、loss、target、law、価格policyは設定から引き継ぐ。
  epsilonの変更に古いQ3適合を流用できるかはprogramの検査に任せる。
- 実行予算・科学条件が未指定なら採択済み設定を調べ、足りない重要事項だけをまとめて確認する。
  既に許可された有限範囲への承認を取り直さない。Skill起動だけで科学実行許可を拡張しない。
  **LDや各roundのshot数を個別に選ばせる工程は設けない。** これらはprogramが扱う探索変数である。
  利用者が範囲を明示した場合はその制約として渡す。

H4の設定・精度・探索範囲を、別Hamiltonianの既定値として採用しない。
正式な入力形式・必須field・予算の単位は公開programのschema/helpに従い、独自schemaを作らない。

## 共通プログラムを使う

1. `program-entry.md`に記録した公開SOURCE、実物の入口、help、問題設定を照合する。
   コマンド・引数はその公開版のものを使い、未対応引数を補わない。
2. programの既存機能で保存結果・進行中jobとの重複を照合し、使える証拠を再利用する。
   SOURCE・physical identity・law・compiler/lowering・測定付きfull wrapper・P=0を保つ。
3. 今回のモードと予算をその入口へ明示する。保存結果モードでは、新規Q3・signal・sampling・
   compile・fitが起動しないことを、programの契約と実績counterで確認する。
   「新規Q3=0」だけで他の科学処理も0になるとは推測しない。
4. 新規科学処理は、採択済み条件と利用者が許可した有限予算内に限る。
   未採択条件はprogramへ適当に埋めず、未評価/不足として返す。
   programの停止理由・予算上限・失敗後のretry/resume条件を尊重する。
5. 再開はprogramの正式な再開入口と状態を使い、別runとして勝手に再投入しない。
   保留や予算到達を理由に、Skillで候補・配分・精度・上限を変更しない。

実行中はprogramのstage/進捗を観測し、SOURCEと入力・出力先を識別できるようにする。
CPU稼働だけを前進や数値gate成立の証拠にせず、programの原結果を保全する。
非0終了は理由と部分結果を報告し、途中で書かれたsummaryだけでrun成功と扱わない。

## 結果を返す

まず「どの範囲まで比較できたか」「何が不足/保留か」「停止理由」を短く述べ、
programの方式別summaryから次を1方式1行で提示する。

| 方式 | 最安適合構成 | shot配分 X/Y | MSE区間・判定 | 総compiled RZ | 価格の資格 | 停止理由 |
|---|---|---|---|---|---|---|

構成はprogramが返すIDとformula/幅/LD/R等、配分は全roundの整数列またはその固定artifactへのリンク。
費用のexact値と表示値、結果SOURCE・run ID・再利用件数/新規call数・探索範囲・保留候補も示す。
該当点が無い方式はUNKNOWN/未評価/非適合を区別し、欠測費用を0にしない。
programが出力しない保証や全方式winnerをSkill側で計算・補完しない。

- 「最安」は調べた有限候補/配分内か、安い保留が残る確認済み最安かを明記する。
  大域最適性を付け加えず、heuristicで未探索を省略した場合はその範囲を引き継ぐ。
- 条件付きQ3モデル適合、物理的energy RMSE保証、十分Nは別の資格として扱う。
- 全26軸MODEL参照値の費用、直接価格、条件付き閉包、有限cohort、E[C]・CIを区別する。
  programの結果が条件付き参照費用なら、その名称を保ち正式Gへ昇格しない。
- 不確かさ・SOURCE不一致・未資格・未対応方式を省略せず、元の状態で報告する。

必要なREPORT/HANDOFFはprogramの成果を参照して差分だけ記録する。適用されるmandatory STOPを維持し、
出力を見て追加探索や価格取得へ自動進行しない。

## 利用例

- `$rpe-cost-compare H4/1.00 Å/STO-3G/DF10、sector(2,2)。要求energy RMSEは0.00159360010199040 Ha。保存結果だけで方式別比較を返して。新規科学処理は0件。`
- `$rpe-cost-compare このHamiltonianのSOURCEと要求精度で、採択済み条件・既に許可した有限予算を用いて方式別比較を実行して。LDとshot配分は共通プログラムに選ばせて。`
- `$rpe-cost-compare このrunを、既に許可した残予算内でprogramの正式な再開機能から継続して。`

2つ目・3つ目も、必要な科学条件や残予算が採択されていなければ、それだけを確認する。
