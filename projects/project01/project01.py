
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------


def get_assignment_names(grades):
    '''
    get_assignment_names takes in a dataframe like grades and returns 
    a dictionary with the following structure:

    The keys are the general areas of the syllabus: lab, project, 
    midterm, final, disc, checkpoint

    The values are lists that contain the assignment names of that type. 
    For example the lab assignments all have names of the form labXX where XX 
    is a zero-padded two digit number. See the doctests for more details.    

    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> names = get_assignment_names(grades)
    >>> set(names.keys()) == {'lab', 'project', 'midterm', 'final', 'disc', 'checkpoint'}
    True
    >>> names['final'] == ['Final']
    True
    >>> 'project02' in names['project']
    True
    '''
    
    dic = {}
    try: # Extract labs
        lab = extract_values(grades, '^lab[0-9]{2}$')
        dic.update({'lab':lab})
    except:
        pass

    try: # Extract projects
        project = extract_values(grades, '^project[0-9]{2}$')
        dic.update({'project':project})
    except:
        pass

    try: # Extract midterm
        midterm = extract_values(grades, '^Midterm$')
        dic.update({'midterm':midterm})
    except:
        pass

    try: # Extract final
        final = extract_values(grades, '^Final$')
        dic.update({'final':final})
    except:
        pass

    try: # Extract discussions
        disc = extract_values(grades, '^discussion[0-9]{2}$')
        dic.update({'disc':disc})
    except:
        pass

    try: # Extract checkpoints
        checkpoint = extract_values(grades, '^project[0-9]{2}_checkpoint[0-9]{2}$')
        dic.update({'checkpoint':checkpoint})
    except:
        pass
    
    return dic

# Helper function to extract corresponding column values from df
def extract_values(df, pat):
    """
    Extracts pat from df keys, and returns
    a list of strings corresponding to pat
    
    :param df: dataframe to extract keys from
    :param pat: pattern to fit
    :return: a list of values corresponding to pat
    """
    
    return list(df.columns[df.columns.str.contains(pat)])


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def projects_total(grades):
    '''
    projects_total that takes in grades and computes the total project grade
    for the quarter according to the syllabus. 
    The output Series should contain values between 0 and 1.
    
    :Example:
    >>> grades_fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(grades_fp)
    >>> out = projects_total(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    '''
    
    grades_mod = grades.fillna(0) # Fill NaN with 0, deep copy
    projects = extract_values(grades, '^project[0-9]{2}$') # Project scores
    free_resp = extract_values(grades, '^project[0-9]{2}_free_response$') # Free response scores

    total = []
    for project in projects: # Loop through each project
        if (project + '_free_response') in free_resp: # If project has free response
            total.append((grades_mod[project] + grades_mod[project + '_free_response'])
                         / (grades_mod[project + ' - Max Points'] + grades_mod[project + '_free_response - Max Points']))
        else: # If does not have free response
            total.append(grades_mod[project] / grades_mod[project + ' - Max Points'])

    return pd.Series(np.sum(np.array(total) / len(projects), axis=0)) # Calculate total project score


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def last_minute_submissions(grades):
    """
    last_minute_submissions takes in the dataframe 
    grades and a Series indexed by lab assignment that 
    contains the number of submissions that were turned 
    in on time by the student, yet marked 'late' by Gradescope.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = last_minute_submissions(grades)
    >>> isinstance(out, pd.Series)
    True
    >>> np.all(out.index == ['lab0%d' % d for d in range(1,10)])
    True
    >>> (out > 0).sum()
    8
    """
    
    threshold = 8 * 60 * 60 # 8 hrs
    lab_late = extract_values(grades, '^lab[0-9]{2} - Lateness') # Lab lateness columns
    grades_mod = grades[lab_late] # Dataframe containing lab lateness columns
    grades_mod = grades_mod.apply(string_to_seconds) # Submission later than threshold
    counts = np.sum(grades_mod[grades_mod != 0] < threshold, axis=0) # Number of late submission
    counts.index = counts.index.str.slice(0, 5) # Slice index
    return counts

# Helper function to extract seconds from string
def string_to_seconds(series):
    """
    Converts string to seconds
    
    :param series: string to split by '：'
    :return: seeconds corresponds to the string
    """
    
    lst = np.array(series.str.split(':')) # Split hour, minute, second by ':'
    return np.array([int(elem[0]) * 3600 + int(elem[1]) * 60 + int(elem[2]) for elem in lst]) # Return seconds


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------

def lateness_penalty(col):
    """
    lateness_penalty takes in a 'lateness' column and returns 
    a column of penalties according to the syllabus.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> col = pd.read_csv(fp)['lab01 - Lateness (H:M:S)']
    >>> out = lateness_penalty(col)
    >>> isinstance(out, pd.Series)
    True
    >>> set(out.unique()) <= {1.0, 0.9, 0.8, 0.5}
    True
    """
    
    penalty_v = np.vectorize(penalty)
    return pd.Series(penalty_v(string_to_seconds(col)))

# Helper function to calculate individual penalty
def penalty(late):
    """
    Calculate penalty of true lateness
    
    :param late: time of lateness
    :return: corresponding penalty
    """

    threshold = 8 * 60 * 60 # 8 hrs
    one_week = 7 * 24 * 60 * 60 # One week = 0.9
    two_weeks = 14 * 24 * 60 * 60 # Two weeks = 0.8, Beyond = 0.5
    
    if late < threshold: # Submitted on time
        return 1.0
    elif late < one_week: # Within one week, 10%
        return 0.9
    elif late < two_weeks: # Within two weeks, 20%
        return 0.8
    else: # Two weeks and beyond, 50%
        return 0.5


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def process_labs(grades):
    """
    process_labs that takes in a dataframe like grades and returns
    a dataframe of processed lab scores. The output should:
      * share the same index as grades,
      * have columns given by the lab assignment names (e.g. lab01,...lab10)
      * have values representing the lab grades for each assignment, 
        adjusted for Lateness and scaled to a score between 0 and 1.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = process_labs(grades)
    >>> out.columns.tolist() == ['lab%02d' % x for x in range(1,10)]
    True
    >>> np.all((0.65 <= out.mean()) & (out.mean() <= 0.90))
    True
    """
    
    all_labs = extract_values(grades, '^lab[0-9]{2}') # All labs columns
    labs = extract_values(grades, '^lab[0-9]{2}$') # Labs columns
    lab_late = extract_values(grades, '^lab[0-9]{2} - Lateness') # Lab lateness columns
    
    grades_mod = grades[all_labs].fillna(0) # Fill NaN with 0, deep copy
    penalty = grades_mod[lab_late].apply(lateness_penalty).rename(columns=lambda col: col[0:5] + ' - Penalty') # Create penalty columns
    grades_mod = pd.concat([grades_mod, penalty], axis=1) # Append dataframes
    
    df = pd.DataFrame() # Create new dataframe to return
    for lab in labs: # Loop through each lab
        df[lab] = grades_mod[lab] * grades_mod[lab + ' - Penalty'] / grades_mod[lab + ' - Max Points']

    return df


# ---------------------------------------------------------------------
# Question #6
# ---------------------------------------------------------------------

def lab_total(processed):
    """
    lab_total takes in dataframe of processed assignments (like the output of 
    Question 5) and computes the total lab grade for each student according to
    the syllabus (returning a Series). 
    
    Your answers should be proportions between 0 and 1.

    :Example:
    >>> cols = 'lab01 lab02 lab03'.split()
    >>> processed = pd.DataFrame([[0.2, 0.90, 1.0]], index=[0], columns=cols)
    >>> np.isclose(lab_total(processed), 0.95).all()
    True
    """
    
    min_score = processed.min(axis=1) # Get minimum score
    return (np.sum(processed, axis=1) - min_score) / (len(processed.columns) - 1) # Calculate total lab score


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def total_points(grades):
    """
    total_points takes in grades and returns the final
    course grades according to the syllabus. Course grades
    should be proportions between zero and one.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """
    
    lab_tot = lab_total(process_labs(grades)) # Labs total
    proj_tot = projects_total(grades) # Projects total
    chpt_tot = other_total(grades, 'checkpoint') # Checkpoints total
    disc_tot = other_total(grades, 'disc') # Discussions total
    mid = other_total(grades, 'midterm') # Midterm
    fin = other_total(grades, 'final') # Final
    total = lab_tot * 0.2 + proj_tot * 0.3 + chpt_tot * 0.025 + disc_tot * 0.025 + mid * 0.15 + fin * 0.3
    return total

# Helper function to calculate disc, checkpoint & exams scores
def other_total(grades, name):
    """
    Given the dataframe and the area name, calculate
    the total grades for that area.
    
    :param grades: dataframe to process
    :param name: area to process grades
    :return: a Series of total area grades
    """
    
    names = get_assignment_names(grades) # Get names
    area = names.get(name) # Get cols of name
    
    df = pd.DataFrame()
    for ar in area: # Loop through each name
        df[ar] = grades[ar] / grades[ar + ' - Max Points']
    
    total = (np.sum(df, axis=1)) / (len(df.columns)) # Calculate total score
    return total

def final_grades(total):
    """
    final_grades takes in the final course grades
    as above and returns a Series of letter grades
    given by the standard cutoffs.

    :Example:
    >>> out = final_grades(pd.Series([0.92, 0.81, 0.41]))
    >>> np.all(out == ['A', 'B', 'F'])
    True
    """

    return total.apply(score_to_grades)

# Helper function to convert scores to letter grades
def score_to_grades(score):
    """
    Given the total score of a student, calculate the 
    corresponding letter grade.
    
    :param score: total score to convert
    :return: convert total score to letter grade
    """
    
    if score >= 0.90:
        return 'A'
    elif score >= 0.80:
        return 'B'
    elif score >= 0.70:
        return 'C'
    elif score >= 0.60:
        return 'D'
    else:
        return 'F'

def letter_proportions(grades):
    """
    letter_proportions takes in the dataframe grades 
    and outputs a Series that contains the proportion
    of the class that received each grade.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = letter_proportions(grades)
    >>> np.all(out.index == ['B', 'C', 'A', 'D', 'F'])
    True
    >>> out.sum() == 1.0
    True
    """
    
    total = total_points(grades) # Calculate total scores
    letter = final_grades(total) # Convert scores to grades
    return letter.value_counts() / len(letter) # Count occurrence


# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def simulate_pval(grades, N):
    """
    simulate_pval takes in the number of
    simulations N and grades and returns
    the likelihood that the grade of sophomores
    was no better on average than the class
    as a whole (i.e. calculate the p-value).

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = simulate_pval(grades, 100)
    >>> 0 <= out <= 0.1
    True
    """

    return None


# ---------------------------------------------------------------------
# Question # 9
# ---------------------------------------------------------------------


def total_points_with_noise(grades):
    """
    total_points_with_noise takes in a dataframe like grades, 
    adds noise to the assignments as described in notebook, and returns
    the total scores of each student calculated with noisy grades.

    :Example:
    >>> fp = os.path.join('data', 'grades.csv')
    >>> grades = pd.read_csv(fp)
    >>> out = total_points_with_noise(grades)
    >>> np.all((0 <= out) & (out <= 1))
    True
    >>> 0.7 < out.mean() < 0.9
    True
    """

    return None


# ---------------------------------------------------------------------
# Question #10
# ---------------------------------------------------------------------

def short_answer():
    """
    short_answer returns (hard-coded) answers to the 
    questions listed in the notebook. The answers should be
    given in a list with the same order as questions.

    :Example:
    >>> out = short_answer()
    >>> len(out) == 5
    True
    >>> len(out[2]) == 2
    True
    >>> 50 < out[2][0] < 100
    True
    >>> 0 < out[3] < 1
    True
    >>> isinstance(out[4], bool)
    True
    """

    return None

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_assignment_names'],
    'q02': ['projects_total'],
    'q03': ['last_minute_submissions'],
    'q04': ['lateness_penalty'],
    'q05': ['process_labs'],
    'q06': ['lab_total'],
    'q07': ['total_points', 'final_grades', 'letter_proportions'],
    'q08': ['simulate_pval'],
    'q09': ['total_points_with_noise'],
    'q10': ['short_answer']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
