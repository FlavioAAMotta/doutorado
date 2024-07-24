# time_window representa qual semana será o início da janela
# windowSize representa o tamanho da janela
# stepsToTake representa tamanho do passo na janela

# Exemplo1: time_window = 1, windowSize = 2, stepsToTake = 1
# [1, 3, 3, 5, 3, 5, 5, 6]
# Exemplo2: time_window = 10, windowSize = 4, stepsToTake = 4
# [10, 14, 14, 18, 14, 18, 18, 22]
# Exemplo3: time_window = 10, windowSize = 4, stepsToTake = 2
# [10, 14, 14, 18, 14, 16, 16, 18]
# Exemplo4: time_window = 10, windowSize = 4, stepsToTake = 1
# [10, 14, 14, 18, 14, 15, 15, 16]
# Exemplo5: time_window = 10, windowSize = 1, stepsToTake = 1
# [10, 11, 11, 12, 11, 12, 12, 13]

def getAllWindows(time_window, windowSize, stepsToTake):
    steps = stepsToTake * time_window
    initialTrain, endTrain, initialTrainLabel, endTrainLabel = getPeriodStamps(
        steps, windowSize
    )
    (
        initialEvaluation,
        endEvaluation,
        initialEvaluationLabel,
        endEvaluationLabel,
    ) = getPeriodStamps(initialTrainLabel, windowSize)
    endEvaluationLabel = initialEvaluationLabel + stepsToTake
    return [
        initialTrain,
        endTrain,
        initialTrainLabel,
        endTrainLabel,
        initialEvaluation,
        endEvaluation,
        initialEvaluationLabel,
        endEvaluationLabel,
    ]


def getPeriodStamps(time_window, windowSize):
    firstPeriod, lastPeriod = getPeriodByWindow(time_window, windowSize)
    firstPeriodLabel, lastPeriodLabel = getPeriodByWindow(lastPeriod, windowSize)
    return [firstPeriod, lastPeriod, firstPeriodLabel, lastPeriodLabel]


def getPeriodByWindow(time_window, windowSize):
    first_period_week = time_window
    last_period_week = first_period_week + windowSize
    return [first_period_week, last_period_week]


def calculate_time_total(num_weeks, window_size, steps_to_take):
    if window_size == 1:
        return range(
            6, (num_weeks // steps_to_take) - ((2 * window_size) // steps_to_take)
        )
    return range((num_weeks // steps_to_take) - ((2 * window_size) // steps_to_take))