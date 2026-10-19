# Seed・Bits interface

実行するnamespaceは `ARTIFICIAL_REAL_ADAPTER_V1/arm/axis` のみ。科学候補namespaceは文字列参照だけ。bits()の各呼出しでもdomain/stateを検査するため、domainをscience文字列へ変えるだけでは生成できない。

固定G7 SOURCEのSHA256(UTF8 seed || uint64 big-endian counter)を静的に照合。必要blockを連結し高位width bitを返し、残りをdiscardする。width/counter/value/discarded/累積consumption/failure positionを記録する。1/160/256/257/512/513bitの人工試験を行った。invalid bool widthは消費前にSTOP。

global identityはsource/input/scope/claim/run/Mode/task/local-global trialへbindする。Modeはglobal IDに入りseedへ入らない。S/P1、A/B、phase/passは同prefix、armとX/Yは分離。実workerは新しいbits_factoryを通る。

人工Canonicalの2 drawと人工Fullの3/4/5 drawはsynthetic shapeであり、実G7のdegree-dependent draw・conditional law・reject rateの同値ではない。実科学bit列生成0、実fresh IID/消費完全同値はNOT_ESTABLISHED。
