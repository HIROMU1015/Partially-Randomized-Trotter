# H6 native B0+I font-cache startup fix

SOURCE `99abc20a29979f3b6b11d9c7127c1e8cf84c643f`。Matplotlib cold font-cacheの警告Timer threadがthread1制約に抵触した。hash/version固定した表示用font一覧JSONを各run/worker MPLCONFIGDIRへbyte-only配置し、FontManager再構築を避ける。科学/compile/JIT cacheの再利用ではない。

- [診断・限定import確認](font_seed_diagnosis_v1.json)
- [前2回の停止・科学予約0](prior_stop_projection_v1.json)
- [人工49件](artificial_test_result_v1.json)
- [source closure](source_closure_v1.json)
- [主担当最終照合](primary_launch_review_v1.json)
- [最終binding](binding_projection_v1.json)
- [byte/profile](byte_profile_binding_v1.json)

利用者の実行・停止後修正指示の範囲で、fresh gate後に修正済み新one-shotを起動する。元の科学provider/compiler/旧6/最大追加18/AS-RSS/thread1/5sを維持。wallnull/当stage累積CPU24h。独立reviewではなく主担当の照合でsubagents0。元grant/停止/UNKNOWN/carryを保全しreset/refundなし。
