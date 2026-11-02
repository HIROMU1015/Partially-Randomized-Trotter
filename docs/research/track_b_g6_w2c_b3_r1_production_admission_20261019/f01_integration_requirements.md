# F01統合境界

**F01_INTEGRATION_PENDING**。固定指摘はcommit `6576e606d4edcc6c53f095ee5bdf750defa99a42` のtechnical_findings.json。
未修正driverはcommit `e4af6c064db805cbec611bcc8f0ff75a4754d38b`、完全SHA `d7947af16baccbd3e0707fa9b4ffcd26650a6569eb6606ee3c08ec443d1dd7ee`。
今回そのdriverは変更していない。Codex Bのlive worktree/index/lock/process/未commitファイルへアクセスしていない。
公開修正版が存在すると仮定せず、moving remote branchを探して自動取得・merge/rebaseもしていない。

次の統合作業が別途承認されたら、担当Codexが以下を行う。

1. Codex B修正版の公開commit・driver完全bytes/SHAを受領し、固定Git/public blobを照合する。
2. 前trial event誤帰属、axis/trial/inflightとlast committed prefix、digest/stream/committed count、angle/save失敗の来歴を差分と人工回帰で確認する。
3. R1の全binding、24 constructor/prepare3/preflight9、N/seed/law/phase/weight/二pass/出力policyが不変か確認する。
4. `Boundary.enter/admit_once` receipt互換性と`with port:`外側STOP接続を最終host SOURCEとして固定する。
   特にadmission直後・source/load_saved/module load・output mkdir・correctness移行前の失敗でもspent保持とSTOPを確認する。
5. correctness不合格でfull-Nを呼ばない人工テスト、baseline/instrumented失敗、failure/STOP保存失敗とclaim非再利用を一体で確認する。
6. 科学/driver/guard/admission/host/input/adoptionの全完全SHA、failure回帰証拠hash、正式契約digest、最終release commitを更新する。
7. 公開・保全確認後、ユーザーの別途science承認を最後に受ける。実approval/store/claim発行もその承認範囲で行う。

未来driver SHA・回帰証拠hash・正式digest・実承認IDを捏造しない。今回のscopeはここへ自動移行せずmandatory STOP。
