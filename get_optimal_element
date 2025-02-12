import csv

# Define the PredictExpression function
def PredictExpression(A, R, alpha, beta, K_2, O_max, min_expression):
    X_E = max(alpha * A - beta * R, min_expression)
    return O_max * X_E / (K_2 + X_E)

# Define the CalculateScore function
def CalculateScore(O_exp, O_sta):
    return abs(O_exp - O_sta)

# Define the main function for optimizing promoter and RBS configurations
def OptimizePromoterRBS(PromoterSet, RBSset, OA_params, threshold, n):
    alpha, beta, K_2, O_max, ARBS_ID, RRBS_ID = OA_params
    
    OutputSet = []
    
    # Loop through all activator and repressor promoter combinations
    for Pi in PromoterSet:
        for Pj in PromoterSet:
            for RBSi in RBSset:
                for RBSj in RBSset:
                    # Calculate the new alpha and beta based on RBS values
                    alpha_new = (alpha * RBSset[RBSi]) / RBSset[ARBS_ID]
                    beta_new = (beta * RBSset[RBSj]) / RBSset[RRBS_ID]
                    
                    # Predict the expressions for both phases
                    O_exp = PredictExpression(Pi[0], Pj[0], alpha_new, beta_new, K_2, O_max, min_expression) # min_expression = 0.01
                    O_sta = PredictExpression(Pi[1], Pj[1], alpha_new, beta_new, K_2, O_max, min_expression)
                    # Filter combinations where OD exceeds the threshold in both phases
                    if PredictOD(Pi, Pj, RBSi, RBSj, OA_params) < threshold:
                        continue
                    
                    # Calculate the score for the current combination
                    Score = CalculateScore(O_exp, O_sta)
                    
                    # Store the results in OutputSet
                    OutputSet.append((O_exp, O_sta, Score, Pi, Pj, RBSi, RBSj))
    
    # Sort the OutputSet by score (descending order)
    OutputSet.sort(key=lambda x: x[2], reverse=True)
    
    # Return the top n best combinations
    BestCombinations = OutputSet[:n]
    return BestCombinations

def read_promoters(file_path):
    PromoterSet = []
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            activator_rpu, repressor_rpu = map(float, row)
            PromoterSet.append((activator_rpu, repressor_rpu))
    return PromoterSet

def read_rbs(file_path):
    RBSset = {}
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rbs_id, rbs_rpu = int(row[0]), float(row[1])
            RBSset[rbs_id] = rbs_rpu
    return RBSset

def read_oa_params(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        # OA parameters are expected in the first row of the CSV file
        alpha, beta, K_2, O_max, ARBS_ID, RRBS_ID = map(float, next(reader))
        return alpha, beta, K_2, O_max, int(ARBS_ID), int(RRBS_ID)

def write_results(file_path, BestCombinations):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['O_exp', 'O_sta', 'Score', 'Activator Promoter', 'Repressor Promoter', 'Activator RBS', 'Repressor RBS'])
        # Write the data
        for combination in BestCombinations:
            writer.writerow(combination)

PromoterSet = read_promoters('./input/promoters.csv')
RBSset = read_rbs('./input/rbs.csv')
OA_params = read_oa_params('./input/oa_params.csv')

BestCombinations = OptimizePromoterRBS(PromoterSet, RBSset, OA_params, threshold, n)

# Output the best combinations to a CSV file
write_results('./output/best_combinations.csv', BestCombinations)
