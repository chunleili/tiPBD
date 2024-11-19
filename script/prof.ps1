python -m cProfile -o profile $args[0]; 
if ($?) { 
    snakeviz profile;
}
