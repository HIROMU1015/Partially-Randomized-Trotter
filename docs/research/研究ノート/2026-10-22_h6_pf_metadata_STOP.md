# H6 metadata追加承認と取得STOP

利用者が63候補の再観測と固定H6後続stageを包括承認。原B0I serializerの不足値を推測せず新観測へ固定した。事前人工14PASS、準備2回のgrant前例外を修正してSOURCE3cff9a4baで一度起動。snapshot_restore1後のscalar shape比較不具合でSTOP、signal/metadata0。入力破損ではない。修正458ea00dd・人工5PASSを保存し、ユーザー指定の失敗時STOPに従って科学再実行はしない。旧H2/H4結果・旧H6 STOPを保全。H6共通域比較は未完了のまま。[結果](../track_a_h6_pf_metadata_STOP_20261022.md)。local commit後mandatory STOP、push0。

## 追記：利用者の「修正を行って」に対応

scalar shape修正にgrant前のstdlib NPY header/dtype/SHA検査を追加し、loader本体の読込経路とH2/H4既存規約を人工試験した。人工26件・保存H6の8配列byte照合PASS。旧STOPを保全して科学再実行0。[修正報告](../track_a_h6_snapshot_loader_repair_20261022.md)。local commitのみ、push0。
