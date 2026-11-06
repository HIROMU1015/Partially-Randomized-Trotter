# SCVの次方向：4つの研究選択肢

2026-11-07 JST。**判断材料の比較。継続・変更・終了、新control、shot数、科学的GO/NO-GOを正式採択しない。**
固定対象・既報数値は[証拠と限界](SCV224_EVIDENCE_AND_LIMITATIONS.md)、候補差分は[候補比較](SCV_CONTROL_CANDIDATE_COMPARISON.md)。
ユーザーが研究方向を選び、必要なら新しい有限scopeを明示許可するまで、Codexは科学実行を開始しない。

## 選択肢と判断を変える証拠

| 選択肢 | 既存証拠からの根拠 | 残る未知・失う研究機会 | 次に必要な最小作業・判断が変わる条件 |
|---|---|---|---|
| 1 現行Sの検証を一区切り | rank/range・接続は確認済みだが、X追加削減なし、Y小さい点差、両軸の学習費込みpoint scoreは不利。自動追加の積極的根拠がない | 現群に小さい真の効果がある可能性、少数trainによる係数損失を解消しない。族全体や他候補は否定しない | 完了結果を固定し、現Sの追加shot/calを保留。明確な別の問いや費用回収の根拠が得られた場合だけ再検討する。追加計算は不要 |
| 2 同じSを限定追加検証 | Y trainのu=0が2件等、不確かさを減らす独立した科学的価値はあり得る | 真の相関が小さいなら追加学習も利益を生まない。全費用とrange/cal penaltyが残る | 「凍結係数の評価誤差」と「再学習するpolicyの係数品質」のどちらを問うか先に分ける。結果が変える判断と追加負担が説明できた場合に、新しい有限scopeを設計する。数・seedは未設定 |
| 3 別の既存controlを検討 | signed supportはh1の符号/support構造、product一致はh2/h3のP_l²=Iという、現群とは別の説明情報を使える | 実H4での非定数性・相関・range・全targetへの利益は未検証。別候補の方が良いとはまだいえない | 候補ごとにknown mean/range/活動層の限定差分と人工テストを具体化。許可後の最小科学段階で、非定数性・独立評価の説明力と全費用を確認する必要がある |
| 4 SCV探索全体を一時停止 | 現在は学習費込み改善の積極的証拠がなく、有限confidence・期待費用への接続にも未完了項目がある | 未検証のsigned support等を含むSCVの機会を当面保留する。一般的無効性の証明にはならない | 現在の成果を保全し、他の既存課題との研究資源配分をユーザーが判断する。新しい課題を本作業から起動しない |

1は**現行Sへの追加取得だけを区切る**選択、4は**別controlを含むSCV探索全体を一時停止する**選択である。
本書はどちらかを実行上の新方針へ登録しない。旧224 campaignのmandatory STOPはどの選択肢でも維持する。

## 同じcontrolを続ける場合、先に分ける疑問

- 保存済みS・λを固定した将来のfresh評価は、その凍結モデルの効果を問う。現在の係数が最適かは答えない。
- 新たな独立trainを伴う検証は、少数出現による係数不安定と、群そのものの弱い相関を区別する材料になり得る。一回の再fitだけで学習policy全体を保証しない。
- 独立calとfresh productionを設計する段階は、精度・成功/返却条件を満たす十分Nと費用を問う。単に今回より多く測ることとは別である。

これらの目的を混ぜて「念のため追加shot」としない。旧評価bitを再fit・群選択へ転用せず、独立性と費用対象を先に決める。
既存学習を固定利用する単発評価では、過去train費と追加費を併記する。再学習するend-to-end policyの費用改善と同一視しない。
旧campaignへのresume/retryや、結果を見ての追加shotではなく、新しい有限契約が必要になる。

## 別controlを続ける場合の最小の問い

**B1-OL後に残るpath依存変動を、安いknown-mean metadataが独立bitから学べる程度に説明し、その効果が学習費・range増を上回る余地があるか。**

既存設計のBorn分解によれば、pathを固定した測定ノイズはmetadata CVで減らせない。
これは検討する科学的問いであり、今回「Bornノイズ主体」と数値判定したものではない。

| 候補 | まず区別したい仮説 | 将来の最小確認内容（今回未実施） |
|---|---|---|
| signed support | fragment IDより細かい符号/support情報が、主にh1で異なる応答を区別するか | 既存catalogのsign/supportをidentity付きで扱う定義、h1の非定数性とexact一位置mean/moment、負値を含むrange、独立train/freeze/paired評価、全費用 |
| product一致 | productの恒等化がh2/h3の応答差に結び付き、全targetで意味を持つか | 一致indicatorの条件付き平均、h0/1/4の定数性、h2/h3活動層のモデル、元全5正層・event phaseの維持、層別加重と有料train |

前者はh1を問えるため現行パイプラインとの差分を限定しやすく、後者は物理的関係が明瞭な一方で活動層を変更する必要がある。
これは実装・問いの違いで、改善量や有望性の数値順位ではない。どちらを採用するかはユーザーの研究判断に戻す。
旧224 bitから候補を再fit・再ランキングする方法は、この最小確認に含めない。

## 有限confidenceと総compiled RZへ進む条件

[af47cf9の費用整理][cost]の連続oracle式

\[
C_a(v)=F_a+K_a/v
\]

は、固定係数・正しい期待単価・連続quota等の仮定を置いた母分散目標の式。
有限confidenceの整数shot保証ではなく、今回のpoint varianceや実現RZを代入してH4のbreak-even値を作らない。

| 分ける段階 | 次の判断に必要な違い |
|---|---|
| oracle可能性 | 真の層内相関があると仮定した削減余地。実測・無料oracleとは違う |
| 有限train | actual凍結係数の品質と失敗/range。rank PASSは精度証明ではない |
| rangeとcal | 独立calでactual残差の分散上界を作る。sample variance=0でも上界0としない。range増大が上界・shot式へ効く |
| 十分整数N・quota | Bernstein/Hoeffding、外向きceil、全正層・minimum取得・失敗予算を含める。点分散の低下とN削減は別 |
| 公平なbaseline | B1-OLにも同じcal・価格情報・誤差配分/配分最適化の機会を与える。固定quota baselineを最適と呼ばない |
| 全費用 | 減る評価full RZが追加train/probe/cal費を上回る必要。cache hitも課金、未知価格/未返却tailを0にしない |
| 期待費用・返却 | 条件付き計画、一attemptの実支出、trainから繰り返すpolicyを区別。STOP/返却・期待単価の保証が必要 |
| 全round RPE | 元H groundの同じenergyへの位相関係、半径・normalization、PF/finite-law/selected bias、全round費と失敗配分の別接続が必要 |

[有限confidence map][confidence]では、SCV用cal dispatch、明示family、層別boundsを持つplan wrapperの最小接続が未完了と整理されている。
今回の完了224はこれらを実施していない。af47の取得前の「224未取得/未採択」という記述を現在の実行状態へ転記せず、**完了状態はba483ab3、保証接続の未完了事項はaf47**から読む。
同じcontrolの点差が少し改善したという理由だけで、独立cal/保証付き比較へ自動移行しない。

## 担当・許可・停止

- **ユーザー**：1–4の継続・変更・保留を正式に判断する。候補・新しいshot数・GOは本書で代行しない。
- **SERVER1 Codex**：選ばれた問いに対し、既存SOURCE再利用の最小差分と有限scopeを具体化する。新control実装・人工実行も今回の静的整理からは開始しない。
- **科学実行**：新metadata平均/moment、追加bit、再fit/保存値の数値再解析、RNG、Born、compile、cal/productionは対象・SOURCE/input・回数・独立性・費用・STOPを定めた別途明示許可が必要。承認証明書/grant発行は要求しない。

並行する可能性があるexact-engine高速化は古典simulation費に関する別作業で、control統計性能や量子RZ改善を代わりに保証しない。
そのSOURCE/worktree/process/cache/成果物は今回調査・変更・操作しない。
本書を含む4 Markdownの公開照合後mandatory STOP。次の科学計算・候補実装・重要科学レビューへ自動進行しない。

[cost]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107/SCV_TOTAL_COST_COMPARISON.md
[confidence]: https://github.com/HIROMU1015/Partially-Randomized-Trotter/blob/af47cf9161241c8710383ff8539a4a7f8f67492c/docs/research/scv_finite_confidence_20261107/SCV_FINITE_CONFIDENCE_MAP.md
