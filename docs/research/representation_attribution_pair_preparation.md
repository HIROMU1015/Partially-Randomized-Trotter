結果前準備記録：constructorのみpreflight最大1866 wrapper、6 A +4 B候補。初回専用testは14 passed /1 failed（testがRTEEvent.probabilityを参照したためAttributeError）。公開field event_probabilityへtestを修正。science runはまだ0。旧library/sampler/source/result不変。

修正後focused testsは85 passed、18既存warnings、fail/skip0。新規専用15件＋既存70件。本batch用全compileはまだ0、test内のsynthetic位相compile6件はscienceと分離する。
