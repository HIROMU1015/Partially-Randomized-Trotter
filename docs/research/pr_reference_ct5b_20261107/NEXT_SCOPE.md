# CT-5B後：必要な最小証拠と承認境界

2026-11-07 JST。[REPORT](REPORT.md)。今回の静的範囲は完了した。以下は次の判断材料であり、採択済みの新科学契約・実行計画ではない。新規grantや承認証明書は作らない。

## 最初の判断

**GPT／ユーザー**が、[構造補題の条件C](STRUCTURAL_CERTIFICATES.md)と記号的価格差の取消しを科学的に採用できるか確認する。これは今回の著者技術調査を、独立詳細科学レビューと呼び替えるものではない。重要レビューを新たに行う場合は、その開始承認も科学計算の承認と区別する。

今回すでに、SOURCEのD固定性、frame pairの共通性、全4 contextの保存配列照合、RTE有限key対応、未知価格の必要条件まで調べた。未解決を理由に独立した静的項目を残してはいない。

## N1：保存価格だけで必要なvacuum境界を数値化する候補

- **目的**：未評価のK、d_o、d_minと、pointwise/期待差に必要なvacuum価格境界を具体的にする。
- **入力**：固定q128/r1/epsilon1e-11、CT-4BのD/CSWAP/scalar anchorと中央4 keyの実価格、V3/CT-0のm7/73-bit保存law・multiplicity、CT-5BのC条件。
- **科学call**：保存データからの限定的な有理数会計・境界評価のみ。新規合成、guard、compile、sampling、simulation、precision/alpha/shot最適化、全18roundは含めない。
- **変わり得る判断**：全support又は期待vacuum capにどの程度の厳しさが必要か。2保存vacuum価格を上界にしない。
- **承認**：**別途の明示的な数値再解析承認が必要**。少数の算術でも今回の§2/§6では未承認であり、CT-5Bでは評価しない。
- **STOP**：有限の指定境界を得た時点、必要確率/価格欠損、compiler条件不成立、対象外の価格評価が必要な時点。改善率を得るために対象を拡張しない。

これはSK実装や大量合成より前に判断を絞る選択肢であり、次作業として正式採択したものではない。

## N2：共有finite domainとevent keyの取得資格を埋める候補

- **未解決**：CT-0のm7/RTE_basis_framesに、CT-4Bにも旧C2-Bにもrecordのない保存keyがある。B′のr1 phase supersetではb2 keyも同一精度recordがない。
- **入力/SOURCE**：保存KEY_COVERAGEのrole_rows/mapping、既存local catalogue、CT-4B PREDECLARED/inventory、同じpygridsynth2.0.0/strict/seed17・epsilon1e-11・actual adjoint規約。
- **最小候補**：新しい角度を設計せず、既存の保存finite key集合から必要な未資格recordだけを事前固定する。b2は、canonicalでの非到達を証明できれば取得対象から除けるが、未観測を非到達と扱わない。
- **科学call**：必要時の限定strict合成・既存guard・record/局所emitter確認。新native compile、全support trajectory列挙、全18round総Tは既定で不要。今回その新実装・key一覧生成・合成を開始しない。
- **変わり得る判断**：共有frameは差から消えても実行に必要、という欠損を有限cacheで埋められるか。共有価格の期待値を新しく求めることが主目的ではない。
- **承認**：実keyと科学callを固定した**別途明示承認**が必要。既存STOPを解除する形ではなく独立領域で扱う。
- **STOP**：指定finite domain完了、guard/phase/adjoint不一致、追加key/precision/別backendが必要。条件変更やSK fallbackへ自動移行しない。

## N3：vacuumの必要境界と実policyを結ぶ候補

**GPT／ユーザー**がpointwise保証か期待保証かを選び、N1等で定まる必要境界に合う証拠だけを求める。選択肢は一様T cap、期待vacuum価格上界、中央/eventとの結合差の安全な上界であり、単独vacuum worst-caseだけに固定しない。

取得/停止保証と価格境界は両方必要。SOURCE recurrenceの有界角度域だけで埋めず、実際に返すwordの価格を覆う必要がある。CT-2/3Aの未資格を維持し、SK初期net/fallback実装を既定作業にしない。既知の取得方法を使える場合も、到達集合・入力角/precision・全域停止・strict guard・word選択capの適用範囲を先に限定する。

新規合成・数値期待評価・新policy実装・科学simulation等を含める場合は、入力・科学call・比較条件・STOPを明示して**別途承認**を得る。CT-5Bでそのアルゴリズムを正式採択したり、数値条件を追加したりしていない。

## 終了と担当

**Codex**：CT-5Bの指定静的成果を公開しmandatory STOP。追加科学実行なし。

**GPT／ユーザー**：条件付き構造導出を評価し、必要ならN1/N2/N3のうち判断を変える最小scopeを選ぶ。既存の同精度・成功条件・B′/Reference公平性を維持し、全18round/T-depth/別Trackへ自動拡張しない。
