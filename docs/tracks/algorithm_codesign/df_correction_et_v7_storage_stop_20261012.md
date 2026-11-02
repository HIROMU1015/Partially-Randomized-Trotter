# E/T v7：全登録計算完了後の集計保存停止

S7 `74682c40d0cdcca92d244a9edf4bb2852361eef5` → A7 `84562f8c5af9249a902662324200456498a83d15`。
E_NATIVEとT_KEYSを完了。native22、T key39（pygridsynth33、exact Clifford6）、retry0。
全key error guardとinventory最終integrityは成功。
末尾のT_accounting保存でValueError: Review file exceeds 512 KiBとなり、原分類はTECHNICAL_INCONCLUSIVE。
原result/marker/terminalと外側exit2を保持する。

大きなunknown_multiplicities辞書がtyped snapshotへ展開され512KiBを超える保存不具合。
次はその辞書をimmutable blobへ分割するstorage修正と、同じsourceのpost-T会計を保存値から復元する。
native/Tの必要計算は完了済みなので追加science/native/synthesisは不要。
原one-shotをCOMPLETEへ書き換えず、独立したsaved-value recoveryとしてprovenanceを保存する。
利用者の技術継続認可に基づき、固定比較条件・guarantee・capsを変更しない。
科学的勝敗はGPT判断であり、この段階では採択しない。
