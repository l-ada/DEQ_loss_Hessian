
1. check if there is a term in gradient updated thtat causes trace to grow?
	This would be something like $+tr(J), e^T J e$ or something similar
2. Fixed point correction points to some stability correction but why?
	Does it approximate BPTT
3. Can I look at simple example $Ax=x, Ax=0$ etc to see if training dyynamics change Jacobian.

4. Selective weight correction : Giving big penalties hurts performance, but helps stability. What iI make big penalties but only for some subset of weights? Would that help?
