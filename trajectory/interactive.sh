#!/bin/sh
srun --account=education-ae-msc-ae --job-name="interactive" --partition=compute --time=00:30:00 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=1GB --pty bash