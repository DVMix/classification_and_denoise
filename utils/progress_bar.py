import sys

toolbar_width = 25

def pr_bar(epoch, i, n, toolbar_width, metrics):
    step = n / toolbar_width
    progress = int(i // step)
    eqas = "=" * progress
    mins = "-" * (toolbar_width - progress)
    stroka = ': [' + eqas + mins + ']'
    percent = round(100*(i/n),2)
    #t = "\rEpoch {}: {}: {}% ({}/{}), {}: {}".format(epoch, stroka, percent, i ,n, list(metrics.keys())[0], list(metrics.values())[0])
    t = "\rEpoch {}:".format(epoch)
    t+= " {}:".format(stroka)
    t+= " {}%:".format(percent)
    t+= " ({}/{}):".format(i,n)
    for key in metrics.keys():
        t+= " {}:".format(key)
        t+= " {}\t".format(round(metrics[key].item(),3))
    
    sys.stdout.write(t)
    sys.stdout.flush()