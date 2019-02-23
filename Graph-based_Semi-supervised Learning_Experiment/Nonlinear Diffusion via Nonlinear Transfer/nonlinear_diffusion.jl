function run_nonlinear_diffusion(pinvDG, NF, pinvDadj, Nadj, n, nclass, sigma, w, idx_train, y, p1, p2, h)
	# Get train set.
	class_train_samples = zeros(Int, nclass, n+1)
	#First entry will be the size of training sampe
	for j = 1:size(idx_train, 1)
		j_index = idx_train[j]
		j_label = y[j_index]
		location = class_train_samples[j_label][1]
		class_train_samples[j_label, location+2] = j_index
		class_train_samples[j_label, 1] = class_train_samples[j_label, 1] + 1
	end
	preds = zeros(Float64, nclass, n)
	# Start diffusion
	   for j = 1:nclass
		@printf("Run nonlinear diffusion for class %f\n", j)
		uF = sparsevec(class_train_samples[j, 2:(class_train_samples[j, 1]+1)], ones(class_train_samples[j, 1]) / (1.0 * class_train_samples[j, 1]), n, +)
		uadj = sparsevec(class_train_samples[j, 2:(class_train_samples[j, 1]+1)], ones(class_train_samples[j, 1]) / (1.0 * class_train_samples[j, 1]), n, +)
		for tt = 1:t
			diff_uF = NF * pinvDG * uF;
			uF = uF - h * NF' * ((abs.(diff_uF)).^(p1-1) .* sign.(diff_uF)) - sigma * (uF - uadj)
			diff_uadj = Nadj * pinvDadj * uadj;
			uadj = uadj - h * Nadj' * ((abs.(diff_uadj)).^(p2-1) .* sign.(diff_uadj)) - sigma * (uadj - uF)
			uF[uF .< 0] = 0.
			uF[uF .> 1] = 1.		
			uadj[uadj .< 0] = 0.
			uadj[uadj .> 1] = 1.		
		end
		f = w * pinvDG * uF + (1-w) * pinvDadj * uadj
		preds[j,:] = max.(preds[j,:], f)
	end
	return preds
end