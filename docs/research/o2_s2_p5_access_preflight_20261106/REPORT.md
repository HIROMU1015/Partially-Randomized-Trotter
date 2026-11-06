# O2-S2 P5 access preflight：報告

2026-11-06 JST。基点：[2b71117ae0f6727cfc9b4968bb6fd96be7265a60][BASE]。専用branch：`codex/o2-s2-p5-access-preflight-20261106`。

**判定は `PARTIALLY_READY`。** P5は有限table上で価格・実proposal・weight・normalizerを接続する技術入力として使える。一方、非列挙の価格依存whole-path samplingと分子PR＋RPE総費用の科学検証を、そのまま実行できる状態ではない。

成果物は [S2_P5_ACCESS_PREFLIGHT.md](S2_P5_ACCESS_PREFLIGHT.md) と本REPORTの2件。科学的GO・新規性・正式採択は判定していない。

## 1. 確認範囲と準備案との整合性

LOCALの直前 `NEXT_STEP.md` を取得して読み、今回指定されたP5と照合した。草案の対象未選択を、今回指示がP5 developmentに限定した。元H厳密groundという最終目的は維持し、P5/000をその代用にはしない。草案のB²単純移植の障害、平均価格とpath価格の違い、旧停止の保全も維持する。

適用AGENTS/PROJECT_MAP、利用可能な共通指示2026-10-25の指定節、前段O2の4文書、固定3571477…統合・e8b19ff…文献調査の関連証拠を再利用した。研究概要・規範・statusの読み方は前段の読了内容を継承し、今回の結果として書き換えない。

新たに実読した中心資料は、G9 v2契約/結果、G10 v3結果とprimary-data access、G7/Closed P5/Fullのgenerator、native/provider、G8 cacheと列挙IS診断、Parentのexplicit-table IS/統計、D1会計とD2境界kernel/D3後続、主RPEのshot/価格接口、関連test SOURCEである。固定commit・path・関数はpreflight本文へ付した。

## 2. 取得できた情報

| 項目 | 今回確認した事実 |
|---|---|
| 固定入力 | p=(1/5,3/10,1/2)、x=5/7、m=5、既知3-qubit synthetic provider。分子geometry/basis/DF rank/L_D/δ窓は非該当 |
| 価格とidentity | G10 accessのP5行には、eventの係数・実proposal・weight、raw/reduced word、child/phase/angle、native IR、T/CX/1Q価格が同じbindingにある。m5はG9の再利用で、独立価格取得ではない |
| 取得粒度 | G9 rawの固定Git object存在、G10のindex/part/exact table/cacheを確認。P5 row4 indexと全63 binding partはGitHubからも取得し、固定blob IDとの一致を確認。他行は固定保存metadata/選択bindingを読解。全raw再構成・全ファイル再監査はしない |
| law/normalizer | 元digital P₀とã/P₀、Closedの群root/DP、Fullのordinary envelope/local packet/zero、既知table ISのroot/dyadic規則を区別できる。理想係数・実確率・保存coefficient normを混同しない |
| 統計 | 共通biasとM2/rangeによる既存Bernstein、主RPEのHoeffding/半径接口がある。元weighted lawの変更にはP₀/qだけでなく元weightが必要 |
| 既存の非加法処理 | event内literal相殺の保存IRに加え、D2/D3にはstep境界相殺kernelと完了結果がある。未実装扱いに戻さない。ただし個別pair表/価格依存normalizerが既に保存されたという意味ではない |

## 3. 主要な不足と技術分類

| 対象 | 準備状況 | 残るもの |
|---|---|---|
| 既存1 P5 trialの明示tableによる既知IS/access診断の設計 | `READY_FOR_SCIENCE_DESIGN` | 次に判断したい問い、主指標・価格境界・共通confidenceの採択。既知ISの再計算だけを独立新機構にしない |
| 非列挙whole-path価格を使うS2 | `PARTIALLY_READY` | path範囲/長さ/r/時刻、価格依存normalizer、実生成経路、取得・online費、wrapper接続 |
| P5単独による分子PR＋RPE総費用の直接検証 | `UNSUITABLE_INPUT` | 物理H/state/DF/PF/D halves、全round・半径・branch/energy成功条件がない |
| 新候補の利益 | `UNRESOLVED` | 新しい科学的なN/価格評価をしていないため、改善・順位は未判定 |

単一event価格、accepted-only価格、全trialの無条件価格、full interrogation価格は別である。準備・readoutの元affine parameterを、状態準備除外の理由だけで全て0へ置き換えない。D2のr=2には別microstep条件があり、無断でP5 x=5/7の反復へ切り替えない。

## 4. 統計・強い対照・負材料

同じ補正target・[-B,B]・残余精度・αなら、既存B²-BernsteinはHoeffdingに正のrange項を加えるため、単純移植では十分shot数を減らせない。より鋭いvarianceを使うなら、その情報・既知保証と取得費が必要である。

既知 `p/√c` はcost×momentの最適化で、range/ceil/全生成費を含む最適法ではない。[原著 §II][IS]。G8には列挙RZ価格IS診断、Parent/S4にはnative tableの既済比較があるため、同じ処理の再発見を次の研究価値にしない。Closed P5とFullの同理想ensemble、CTSのzero-T/小CX、Parent終了条件を維持し、全方式に同じ情報・統計評価・cache/相殺機会を与える。

## 5. 次の最小検証案と担当

ユーザー／GPTが、P5を用いた次の問いを選ぶ。

- 一trial内のword価格で既知ISのアクセス接続を確認するだけなら、既存binding/table/weightと共通有限confidenceを使う限定案にする。G8/Parentとの重複後に、何を新たに確認する必要があるかを先に定める。
- 複数event境界を利用するS2なら、pathの意味・有限target・時刻/rを先に指定し、既存boundary kernel/cacheのどの情報を再利用するかを固定する。個別価格やnormalizerを新たに評価する処理は別科学実行である。

新しい精度、成功確率、主資源指標、sampler仕様は未決定のまま。Codexは必要な範囲が選ばれた場合だけ、実行前条件を具体化する。実際の再会計・sampling・回路/価格評価・実装はそれぞれ明示された範囲の許可後に行う。重要科学レビューも独立した開始許可が必要。

同情報既知法への還元だけ、weight/range/取得費による利益消失、取得不能なnormalizer、同target/full価格の不成立はその具体案の終了材料とする。改善率・性能順位・科学GOの閾値は今回設定しない。

## 6. 文書検査・公開・保全・STOP

変更は新directoryの2 Markdownだけ。既存科学branch/main、SOURCE、raw/results、契約/grant/marker/STOP、前段O2公開4文書、LOCAL草案は変更しない。新しい科学結果・正式研究判断・evidence statusを発行していないため、研究概要・manifest・既存索引は更新しない。本REPORTを入口とする。

既存Phase 2A文書checkerの標準ライブラリだけの `git/blob_oid` を再利用し、UTF-8・改行・相対link/参照定義、30文書linkと26固定Git path、非公開path/認証情報の混入を検査してPASSした。m5六方式のindex/宣言partと選択bindingのschemaも確認した。確率・moment・価格・Nの再計算や旧科学監査の再実行ではない。恒久validator・新台帳・承認管理物は作らない。専用branchへcommit/push後、remote SHAと両成果物の取得内容を照合し、その結果は最終チャットに記す。

新科学計算、保存値の再会計/集計/ランキング/予測、sampling/RNG replay、H/state/reference生成、build/compile/native synthesis、新sampler/算法実装、科学test、重要科学レビューは未実施。Pythonは標準ライブラリで保存JSONの構造・pointer・fieldを読んだだけで、科学SOURCEをimport/実行していない。

**公開・取得照合でmandatory STOP。** P5から別入力へ切り替えず、C1凍結やParent/D/G系列のSTOPを解除しない。旧one-shotのretry/reuse、次研究フェーズへの自動進行は行わない。

[BASE]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/commit/2b71117ae0f6727cfc9b4968bb6fd96be7265a60
[IS]: https://arxiv.org/html/2603.13495v1#S2
