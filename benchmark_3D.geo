// Gmsh project created on Wed Oct 12 11:22:56 2022
SetFactory("OpenCASCADE");
//+
Box(1) = {0, 0, 0, 2.2, 0.41, 0.41};
//+
Sphere(2) = {0.2, 0.2, 0.2, 0.05, -Pi/2, Pi/2, 2*Pi};
//+
BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
Physical Volume("fluid", 28) = {3};
//+
Physical Volume("solid", 29) = {2};
//+
Physical Surface("inlet", 30) = {8};
//+
Physical Surface("outlet", 31) = {13};
//+
Physical Surface("walls", 32) = {11, 12, 9, 10};
//+
Physical Surface("obstacle", 33) = {7};
