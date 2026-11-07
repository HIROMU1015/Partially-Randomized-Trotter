# N4-A：既存辞書による次の最小実行scope案

2026-11-08 JST。[REPORT](REPORT.md)。**提案・未実行。今回の許可を拡張するものではない。** N4-A完了でmandatory STOPし、次担当をGPT／ユーザーへ戻す。

## 1. 固定入力

次に実行するなら、本N4-A公開commitと以下を固定する。

- 変更しないN3-B1b-1の43基本word、C0全299 word、b36／j=-36…6／基本10^-13／最終native10^-11。
- [ROLE_COVERAGE.json](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/ROLE_COVERAGE.json)のnon_C0_domain_candidate_keys **85個**。内訳は共通frame84、B′ b2が1。追加keyを自動生成・追加しない。
- 指定frame `0x1.08ba32d420f2cp+2`、b2 `0x1.921fb54442d18p+2`はその85個の内数。
- SOURCE／compiler A／q128/r1/m7、canonical law、global W／actual adjoint、旧reset/readout、保存199 shots/axis等を維持する。

85は保存finite numeric supersetであって、全vacuum keyの列挙ではない。vacuum全域の既存provider証明はそのまま再利用し、新しい位相分布・到達列挙を行わない。b2の到達性を先に解かなくても安全なsuperset接続を検証できる。

## 2. 明示承認が必要な最小call

**既存providerの再利用のみ**を基本とする。新規基本合成・guard・precision選択は既定scopeに含めない。

1. 85 positive keyに既存provideを適用し、実選択sequence hash／composite certificate／role対応を保存する。負側は同じpositive word全体のactual adjointへ対応させる。これは今回未許可だった新role実行であり、別途の明示承認が必要。
2. 必要最小限の局所emission・operator診断を、指定2 key、保存basis pair、C0とmissの混在、正負・zero・phase境界で行う。実guardは既存基本recordを再利用し、新しいbackend wordを取得しない。数値toleranceをcertificateにしない。
3. 共通lookupとunitary segment consumerを薄く接続し、同一positive keyに同一word、負角全体inverse、inplace、reset/clbit/readout/global phase保持を検証する。
4. 全物理位置のstrict誤差が保存の最終gate要求へ接続することを、SOURCE・certificate・prefix phase処理で確認する。

全85に対する巨大行列計算やexpanded whole CT回路のbuild／simulationは、word存在証明の前提ではない。必要な局所試験だけを実装し、既存C0資格・N3全域証明を理由なく全面再実行しない。新T期待値／極値／全回路会計もこの最小scopeへ自動追加しない。

## 3. 固定できた局所target

[INTEGRATION_TARGETS.json](../../../artifacts/pr_reference_n4a_coverage/2026-11-08/INTEGRATION_TARGETS.json)は対象archive member・record hashを保存している。

| 用途 | 保存入力 |
|---|---|
| B′ basis pair | Bprime_local_catalogue.tar.xz の catalogue/rare1_Y1.json、request=[true,true] |
| Ref basis pair | local_catalogue_boundary.tar.xz の catalogue_boundary/reference_ccx1_rare1_Y1.json、request=["reference",true,true,true] |
| 共通basis id | df-basis-cb0c4fc6dd74e883da97770cd31022dd8e2e647befe0f44f88d4018e214391e2 |
| record名 | basis:上記id、basis_inverse:上記id |
| Pのliteral binding | 両memberのp_symbolic。保存templateはRZ(symbolic_correction)1個 |

basis_inverse recordに指定frame keyがある。新compileでこれを取得し直す必要はない。局所native gate productの置換試験とSOURCE unitary再構成を区別し、未保存global phaseを0補完しない。

whole consumerのmetadata参照は保存CT-4B NATIVE_INPUT_CHECKSのB′/Ref×X/Y、rare0/1の8行。現在のqubit／operations／global_phase_hex／QPY hashを同JSONへ引用した。新しいtrajectoryを生成してscopeを拡大しない。

## 4. Compiler Cで残す条件

現時点で新gate familyは見つかっておらず、C下の保存templateを使う経路が候補となる。既存4-context監査をやり直すことを既定条件にしない。

必要なのは、実consumerが同じpost-unroll literalとSOURCE inverseを保ち、word置換をcompiler後へ置くことの確認。symbolic Pへ実binary引数を束縛するときも、元nativeのglobal phaseを保持する。

保存contextにないfamily／literal変換／phase情報が必要になった場合は、その具体的な欠損を保存してSTOP。追加local compileを要するなら、gate・引数・context・数量を事前固定して別途承認する。未固定の「必要ならcompile」を実行許可とみなさない。

## 5. q128の誤差／RPE資格へ必要な入力

全support CT RPEを扱う段階では、次を固定して使う。

- CT-0 PRECISION_FEASIBILITY.jsonのm7両arm×X/Y：worst-case native RZ数、SOURCE部品上界、追加raw disk予算、保存N条件。
- CT-4B ERROR_AND_SHOTS.jsonのq128／一様10^-11に関する既存条件式。
- 保存native metadataとreset/readout、SOURCE→native・vacuum算術の既存d_old、元nominal半径とnormalization。
- 現在のword certificateに含まれる量子化・基本guard・emitter。初期phase／segment joinの未包含項があれば別途明示。

全位置で既存の十分条件へ包含できることを証明できれば、新しいshot・alpha最適化は不要。V3の期待多重度やsigned境界補正をworst-case誤差数に使わない。

数値のdisk再会計が必要な場合は、**m7/q128/r1両arm×X/Yだけ**の固定再解析として明示承認を求める。compiler diskと合成diskを同じ単位で一度だけ加え、その後半径／位相へ変換し、保存199・alpha1/720・phase.4・failure.05を検算する。現在はその再解析をしていない。全18roundへ拡張しない。

## 6. 終了・STOP

固定85 role接続と局所検証の完了、域外key／C不適合／phase・adjointの重大不一致、必要証拠の欠損、予算不足、未承認のcompile・guard・追加key・価格／RPE計算が必要ならmandatory STOP。

辞書を拡張したり、C0を安いwordへ変更したり、別seed／backendへ移ったりしない。N2取得runner、SK、18round総Tを自動開始しない。

**GPT／ユーザー**が上記から必要な限定scopeを判断・承認する。**Codex**はその明示範囲に限って実装・局所実行を行う。N4-Aの現在時点では、静的成果公開後に停止する。
