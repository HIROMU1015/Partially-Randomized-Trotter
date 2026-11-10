# 留保読み込みの順序修正

2026-11-10 JST。開発評価SOURCE `43f748655c64756c2c2a21d754870c2f7b321a77`の計数モデル・元runner・counterは変更しない。

初回holdoutは`run_h4_df10_price_v1.py:186`の`ValueError('saved trial binding')`で停止した。HOLDOUT.jsonは未作成。停止までにPF q16のX/YとPR確認path0のX/Yの4予測を計算したが、関数が完了せず返却結果は保存されなかった。この未保存の部分結果をPASS・全留保完了と扱わない。

原因は元`trial_files`の保存順をpath番号と同一視した新runnerの不備である。元リストの先頭は0,3,7,2,1,…で、並列取得の完了順だった。全33本のdirectory番号とresult.indexは一致し、0…32が各1回存在した。candidate、cohort、preparation hash、status、conditionalフラグも元対象条件と一致した。原本の不一致や価格モデルの反証ではない。

必要な修正は**元リストの同じ33参照をpath番号順に並べて渡すことだけ**。`complete_h4_df10_price_v1.py`で新SOURCEとしてこのmetadata接続を固定する。元SOURCEのmodel/runner/counterは全bytesを照合してそのまま再利用し、model_sourceと実際のcompletion execution_sourceを別に記録する。旧DEVELOPMENT.jsonを変更・再計算せず、同じ固定ruleを留保へ適用する。集合・event・価格・式・係数・閾値・正しさ条件を変更しない。

これは承認済みV1内の通常の技術修正・継続で、別条件の科学実行・fit・追加compileではない。初回停止を本記録と旧SOURCEに保持する。失われた4予測だけは継続時に同じ入力・同じruleで再評価される。全体を初回一回で完了したとは報告しない。原本や旧run/STOPへは書き込まない。
