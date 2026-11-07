# O2案M：技術実装・人工correctness報告

2026-11-08 JST／SERVER1 GPUA100-gleap-01。base `154c6e3cf1f48d27f558003b0e9bd74882414d9e`、専用branch `codex/o2-m-instrument-20261108`、独立worktree `/tmp/o2-m-instrument-20261108`。

**共通factorizationの十分条件adapterと、Product/Rotationの動的測定instrumentを実装した。最終人工12件PASS。** 元Hadamardの複素平均を保持し、M2は群平均の式と一致した。強い既知法への優位性は未確立であり、適用可能なancilla-free groupingが案Mより良い人工例も確認した。新しい実H4計算・旧raw数値再解析・実shot/RNG・実compile・重要科学レビューは0。

## 成果物

- [M_ELIGIBILITY_AND_FACTORIZATION.md](M_ELIGIBILITY_AND_FACTORIZATION.md)：同じη/slot/selected policy、A/B/V/角/phase/wires/actual Qと具体的な拒否理由。元targetへの実H4 bindingは未取得。
- [M_INSTRUMENT_AND_ARTIFICIAL_TESTS.md](M_INSTRUMENT_AND_ARTIFICIAL_TESTS.md)：制御0 A†/制御1 B、system破壊測定、if_else、X/Y、重みdecoder、新人工参照と強いbaseline。
- [M_ERROR_AND_RESOURCE_BOUNDARIES.md](M_ERROR_AND_RESOURCE_BOUNDARIES.md)：新しい数値・hardware誤差、full-wrapper/古典費、現時点のUNKNOWN。
- [NEXT_MINIMAL_SCIENCE_DRAFT.md](NEXT_MINIMAL_SCIENCE_DRAFT.md)：固定N1のn=0唯一rotation slot、一回metadata-only passの未採択案。
- 新[module](../../../src/trottertracks/algorithm_codesign/m_label_average.py)、[test](../../../tests/tracks/algorithm_codesign/test_m_label_average.py)、[人工runner](../../../scripts/tracks/algorithm_codesign/run_m_label_average_artificial.py)、[最終SOURCE/結果/log/STOP](../../../artifacts/o2_m_instrument/2026-11-08/artificial_run03/)。

## 技術成果と限界

適格性はliteral gate記述・位相を含む十分条件であり、近い行列やchannel等価性で受理しない。actual整数質量から群massと条件付きFraction qを区別して保存する。人工adapterは外部から正しいSlotLabelを受け取るため、実保存policyとの結合まで自動証明したとはしない。量子buildは1/2 system qubit限定。hardware feed-forwardは未資格、数値誤差上界はUNKNOWN。

新しい独立人工参照は、元label別controlled回路、生成したmeasure/if_else全分岐、解析的M2を照合した。正常化、scalar/event/controlled relative phase、逆順、signed/identity、±φ/0、r=0/m=0、Product→Rotation、wire/bit/Y符号、不適格拒否を確認した。statevector/Bornは人工参照側だけで使い、新algorithmへoracleとして渡さない。旧testは再実行していない。

最終run03は12件0 failure/error、runner 2.5427秒、peak RSS 461,208 KiB。開発run01の完全一致assert失敗、修正後run02 PASS、強いbaseline追加後run03 PASSを別directoryに保全した。最終コードhashはrun03 SOURCEで一致を確認する。これは公開commitに同梱するlocal人工証拠であり、CI/外部再現や実H4の科学的改善ではない。

群平均はbranch型label別直接測定に対してM2を減らし得るが、既知の同じ平均observable groupingとは同一機構である。A=B†ならancilla-free直接測定も比較対象に必要で、人工例ではそちらが同等以上だった。したがって独立新規性・科学的GO、CV一般優位、実H4の分散/N/期待費用改善を採択しない。M2差を元との「不一致」と誤解せず、平均保存とM2の意図した変更を分ける。

## 再利用・保全・次の境界

正式共通指示2026-10-25（SHA256 `e1e3e2625e2b728977402bded6583658e7fb762160f5e787fd45572af0095720`）、適用AGENTS、PROJECT_MAP、研究概要・目的・方法、設計4文書とO2統合を確認した。旧SOURCE/STOPと§8.1/8.2/8.4/10の許可・保全・再利用規則を維持する。

既存`CategoricalLaw`、A2の明示gate検査、Hadamard bit mapping、人工matrix helper、Qiskit inverse/control/if_testを再利用した。RTE/DF/selected basis/partial-S2/価格接口と関連testを静的確認したが、適合するsystem-readout/grouped instrumentは確認範囲に存在しなかったため、新規差分を1 moduleと新test/薄い人工runnerに限定した。旧A2/DF等のcodeは無変更。

元main・dirty/untracked、旧SOURCE・結果・raw・STOP・grant/ledger/marker、O2統合/設計、C1凍結、O1/O3/C-S、他worktree/process/cacheを保全する。共有venv/OS/systemd/cgroup/scheduler等は変更しない。人工処理前の空きメモリ/負荷を確認し、この小規模検査に新しい恣意的resource上限や科学managerを設けていない。

次は実H4のgroup適合という**新しい科学的metadata集計**が必要になる。まず一回・1 slot・metadata-onlyの草案を採否判断する。現在のQiskit人工adapterを実H4へ呼ぶのでなく、回路buildを避ける薄い保存header binderを追加する必要がある。実state/Born/shot/compile・新誤差認証・総費用は後段の別許可対象。

専用branchでcommit/公開し、remote SHA、公開本文、blob/tree、最終実行コードhash、旧資料/main保全を照合する。自己SHA追記のための追加commitは作らず、公開固定commitは最終チャットと作業用照合記録で示す。

公開前の文書/SOURCE技術照合はPASS。新5文書の19参照、最終実行の7 SOURCE hash、旧helper byte不変、manifest旧19項目、元mainのHEAD/statusと既存2,136ファイルbyteの保全を確認した。既存Markdown7点は追記のみ。新artifact以外の旧raw/STOPを編集せず、人工correctness以外の既存suiteを追加実行していない。

**公開照合後mandatory STOP。新しい実H4 metadata評価、Mの科学測定、案D/O1/O3、旧研究再開、重要科学レビューへ自動移行しない。**
