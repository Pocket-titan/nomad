conda remove -n tudat-space -y --all;
conda create -n tudat-space python=3.9.19 -y;
conda activate tudat-space;
conda install tudat-team::tudatpy -y;
conda install tudat-team/label/dev::tudatpy -y;
