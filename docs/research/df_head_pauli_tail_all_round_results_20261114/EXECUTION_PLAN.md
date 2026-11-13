# 承認済みPauli tail残22roundの固定実行経路

ユーザーの今回の明示承認に従い、P1結果e8641b84fd8a3e98a49b65ecaa8fbd371a9d8ab7の未評価22件だけを実行する。新caller/統合helper・人工テストをcommitして新科学SOURCEを固定し、fresh run01を一回起動する。P0/P1のSOURCE・raw・結果・KEEP/STOPは変更しない。

## 条件とcall範囲

保存Linear H4/1.00Å/STO3G、tol-only1e−8 DF10、同H/state/snapshot/保存DF-H target、保存completion/fragment順、S2/ΔPF=.1/K2/h=.1/M12/P=0、元prior/decoder/誤差契約、scalar一回会計を維持する。κはP1に保存したexact有理数をそのまま使用する。

- LD1とLD2で、m=1/2/3/4/5/7/8/9/10/11/12を各11件、**計22 new mean/最大66 complex出力**。
- P1のm0/m6×2prefixの4 meanは元JSONと内容照合後、SOURCE付きでそのまま再利用。signal、normalizer、allowanceを再計算しない。
- 新meanはすべてsector36。新full256 mean、P0 operator/JW辞書再計算、DF mean、reference actionは0。
- unique finite normalizerは新S2 roundあたり最大1組、計最大22組。P1の5組を再計算しない。今回のfresh cacheのcall上限を22にし、旧P1の実行契約・数値gateは変更しない。
- N/nominal N/RPE MSE/decoder/sampling/circuit build/compile/価格/G・候補探索は0。

## SOURCE・identity

P1のB native canonical configurationをJSONごと再利用する。

| prefix | 固定canonical ID |
|---|---|
| LD1 | `d905d2197d4b01df3074b3844c0f4a2b990c21b9c3e1a0e3b6ea13a488b1fd27` |
| LD2 | `d7dd4f52bece343126258ffa24b9c4d7e1b8d76af82d1f4f4004819e832aa4d3` |

B SOURCEはP1と同じce0517de6e7b58cf4bf4c69f99f998d86099c406のcanonical validator。13round identity/lawを照合し、Pauli辞書・λ/scalarのSOURCE/hashを保持する。新IDの作り直しやDF IDの流用はしない。RはP1で共通κから生成済みの各round値を使用する。P1のfixed-R controlを本番configurationへ取り込まない。

数値providerとP1 adapterはe8641b84時点からbyte不変で使う。新SOURCEはcaller/batch/統合部分を固定し、新22recordのnumerical recipeに記録する。旧4recordのsourceは05447d1d4のまま保持する。packet全体のSOURCEだけで全recordを再ラベルせず、各roundのmean_sourceを正本とする。

## 数値処理・STOP

既存P1 loaderがP0のfull256 operatorと保存stateを読み、合計tail/headのsector36行列を供給する。個別Pauli eventを射影unitaryとして扱わない。今回full256で伝播するcallは0であり、P0/P1の全空間検証と今回の縮約ideal meanの役割を分ける。

既存providerのdirect-expm head、paired finite Taylor、独立raw/corrected、exact-tail、scalar、matrix power/cache、R-aware empirical allowance、normalizerの丸め診断を変更せず使用する。head/exact-tailの同じmacro演算はcache共有し、巨大event列を生成しない。

reference pair gate1e−10、state norm1e−12、unitarity/norm/closure1e−9を維持する。長時間のclosureやnormalizer精度が不成立なら、その値を保存して停止する。閾値緩和、再正規化、再試行、候補/κ/K/PF幅の変更や追加科学callはしない。

呼出しはm昇順でLD1/LD2を交互に処理する。最初の失敗で以後のcallを起動しないSTOP契約と、小さい36次元行列のcache再利用を優先するためである。独自のCPU/RAM/worker制限や共有設定の変更は行わない。

## 統合成果物

個別recordとpartial/final resultを新runへ保存する。raw結果をcommitで固定後、そのcommitと旧P1 commitを各roundのlocatorに持つB mean packetを生成する。旧4recordは元値・identity/law・numerical SOURCEを保ち、新22recordは新SOURCEとraw carrierへ結びつける。

成功時は各13roundのsignal packetを完成する。失敗時も未評価・失敗を区別したpartial packetを保存し、数値を補完しない。どちらも十分N/global RMSE/C/G/winnerはnullのままで、全体RPE資格を認定しない。

人工6件と既存P1回帰6件で、22件限定・旧値保全・first-failure STOP・SOURCE付き統合・人工m12を確認する。完了後REPORT/HANDOFFと必要な研究記録をcommit/push・remote SHA/tree/主要成果物照合し、mandatory STOP。
