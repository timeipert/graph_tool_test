
from mtam import mtam

model = mtam()
model.load_model("dc-hsbm.pickle")
model.refine_states()
print("Worked. Minimal Description Length: ", model.mdl)


# My output is on the terminal is:
#
# Run  0
# beta = 1
# 62296.55797811627
# 59304.967547308326
# 58332.356779513095
# 57816.99010279525
#
# Process finished with exit code 139 (interrupted by signal 11: SIGSEGV)