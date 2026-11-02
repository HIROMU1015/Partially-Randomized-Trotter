# H6 B0+I blueprint結合修正

run01停止原因の旧preparation hashと現在profileの不一致を3caseで再現。旧hash保持＋現profile/input/providerに固定した3blueprintの厳密照合、実/期待field差の保存を実装。SOURCE6b4d3026…、新12＋影響70の82人工tests PASS。数学/compiler/最大6並列・合計6compileは維持。サブエージェント0、限定snapshot診断2/preparation6以外のsignal/native build/probe/compile・新grant・本計算再起動0。未認可draftへ再結合し、旧run/carryを返却せず保全。

[詳細](../track_a_h6_b0i_blueprint_fix_20261015.md)。旧thread4の比較でも現thread1と同じhashとなり、thread数を原因と断定しない。旧内部prepared payload差は再構成できないためbit同一性は未認定。旧hashと現在profile期待値を分離して固定し、物理correctness gateを維持する。利用者の指示により単独確認、独立レビュー・新実行承認の代行なし。
