# 研究④：DF全energy共分散候補の方法差検証

ユーザーの限定指示により、調査commit ab0c4d0fの候補とKoizumiほか2606.30601v2 S8、著者SOURCE206aa441を読み、[紙上比較](../qc_iv_covariance_method_audit_20261018/README.md)を作成した。

技術判定は `REDUCES_TO_KNOWN_METHOD`。既知class式へDF因子を代入した対照K*に、同じstate情報・precision・標準縮約順序・streaming/cacheを許せば同じ演算列になる。独立差は残らない。著者CLIのDF全energy実装済み・correctness/runtime確認済みとはせず、1-/2-/mixed covariance、高次相関、紙面とSOURCEの添字規約、impossible chargeのraw momentとcovarianceの違いも記録した。

還元時STOPという今回指示に従い、この候補の探索は終了し、実装・新数値・新保証探索へ進まない。新規性・科学的GO/NO-GO・④の次期方針を正式採択したものではない。A2現経路追加投資見送り、A2 NOT_ADOPTED、A1 CLOSED、K1-F未承認と他の停止状態を保持する。新科学計算・保存値再解析・科学SOURCE実装/実行・科学契約/認可等0、公開・保全確認後mandatory STOP。次はユーザーの研究課題選定への反映判断であり、この候補の追加Codex作業は不要。
