// Gmsh project created on Thu Oct 13 17:01:49 2022
SetFactory("OpenCASCADE");
//+
Box(1) = {0, 0, 0, 2.2, 0.41, 0.41};
//+
Cylinder(2) = {0.2, 0.2, 0.05, 0, 0, 0.3, 0.05, 2*Pi};
//+
BooleanFragments{ Volume{1}; Delete; }{ Volume{2}; Delete; }
//+
Physical Volume("fluid", 28) = {3};
//+
Physical Volume("solid", 29) = {2};
//+
Physical Surface("inlet", 30) = {10};
//+
Physical Surface("outlet", 31) = {15};
//+
Physical Surface("walls", 32) = {11, 12, 13, 14};
//+
Physical Surface("obstacle", 33) = {7, 8, 9};
