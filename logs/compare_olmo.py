import json, sys
cloud = json.load(open("results/results_psi_1b.json"))
orig  = json.load(open("logs/olmo_mps_orig.json"))
fam   = json.load(open("results/results_psi_olmo.json"))["results"]
regs = ["top-10","top-50","top-100","top-1000","plausible","weighted"]
worst_psi = worst_ent = worst_fam = 0
print(f"{'delta/region':<28}{'cloud':>9}{'MPS':>9}{'diff':>9}  {'MPS 95% CI':>20}  {'cloud 95% CI':>20}  verdict")
for d in cloud:
    for r in regs:
        c, m, f = cloud[d][r]["mean"], orig[d][r]["mean"], fam[d][r]["mean"]
        worst_fam = max(worst_fam, abs(m - f)); diff = m - c; worst_psi = max(worst_psi, abs(diff))
        ci, cci = orig[d][r]["ci"], cloud[d][r]["ci"]
        print(f"{d+' '+r:<28}{c:>+9.4f}{m:>+9.4f}{diff:>+9.4f}  [{ci[0]:+.3f},{ci[1]:+.3f}]  [{cci[0]:+.3f},{cci[1]:+.3f}]  {'PASS' if abs(diff)<=0.02 else 'FAIL'}")
    for k in ["ref","tgt","drop"]:
        c, m, f = cloud[d]["entropy"][k], orig[d]["entropy"][k], fam[d]["entropy"][k]
        worst_fam = max(worst_fam, abs(m - f)); diff = m - c; worst_ent = max(worst_ent, abs(diff))
        v = "PASS" if abs(diff)<=0.01 else "ACCEPTABLE" if abs(diff)<=0.05 else "INVESTIGATE"
        print(f"{d+' entropy '+k:<28}{c:>+9.4f}{m:>+9.4f}{diff:>+9.4f}  {'':>20}  {'':>20}  {v}")
print(f"\nmax |psi diff| MPS vs cloud = {worst_psi:.4f}   max |entropy diff| = {worst_ent:.4f}")
print(f"max |diff| new script vs original script on same dumps = {worst_fam:.2e}")
