#!/bin/bash -l
#SBATCH --job-name=CNVs
#SBATCH --partition=cuPartition
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --nodes=1

#Mapping 
ID=$SLURM_ARRAY_TASK_ID
sample=CRR`expr $SLURM_ARRAY_TASK_ID + 246423`
datadir=/MP_reseq/
ref=CaT2T.genome.fasta

mkdir $sample
cd $sample
bwa mem -t 50 -M -A 1 -B 4 -O 6 -E 1 \
   -R '@RG\tID:pep'$ID'\tSM:pep'$ID'\tLB:PRJCA004361\tPL:illumina' \
   $ref ${datadir}${sample}/${sample}_R1.fq ${datadir}${sample}/${sample}_R2.fq \
   > ${sample}_T2T.sam
samtools sort -@ 50 -o ${sample}_T2T.bam ${sample}_T2T.sam
rm ${sample}_T2T.sam
echo "Mapping done"



#CNV calling
ID=$SLURM_ARRAY_TASK_ID
sample=CRR`expr $SLURM_ARRAY_TASK_ID + 246423`
#samples=("53A" "CTJA" "JJA" "LSA" "NJA" "QYA" "TJA" "XJA" "XMLA")
#sample=${samples[$ID-1]}

cat CaT2T.CBGs.list.sort.bed | while read line 
do
region=`echo $line | awk '{print $1 ":" $2 + 1 "-" $3}'`
samtools coverage -H -r $region ../${sample}/${sample}_DH.bam >> covfiles/${sample}.caps.cov
done
