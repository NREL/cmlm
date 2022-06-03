# color palletes from seaborn

def get_color(allinput=None, index=None, color=None, shade=None, string=None):
    
    colors = ['blue', 'orange', 'green', 'red', 'purple',
              'brown', 'pink', 'gray', 'gold', 'aqua']

    color_order = ['blue', 'orange', 'gray', 'green', 'red', 'purple',
                   'gold', 'aqua','brown','pink']
    
    shades = ['d','m','l']
    
    rgbs = [(0.0, 0.10980392156862745, 0.4980392156862745),
            (0.6941176470588235, 0.25098039215686274, 0.050980392156862744),
            (0.07058823529411765, 0.44313725490196076, 0.10980392156862745),
            (0.5490196078431373, 0.03137254901960784, 0.0),
            (0.34901960784313724, 0.11764705882352941, 0.44313725490196076),
            (0.34901960784313724, 0.1843137254901961, 0.050980392156862744),
            (0.6352941176470588, 0.20784313725490197, 0.5098039215686274),
            (0.23529411764705882, 0.23529411764705882, 0.23529411764705882),
            (0.7215686274509804, 0.5215686274509804, 0.0392156862745098),
            (0.0, 0.38823529411764707, 0.4549019607843137),
            (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
            (0.8666666666666667, 0.5176470588235295, 0.3215686274509804),
            (0.3333333333333333, 0.6588235294117647, 0.40784313725490196),
            (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
            (0.5058823529411764, 0.4470588235294118, 0.7019607843137254),
            (0.5764705882352941, 0.47058823529411764, 0.3764705882352941),
            (0.8549019607843137, 0.5450980392156862, 0.7647058823529411),
            (0.5490196078431373, 0.5490196078431373, 0.5490196078431373),
            (0.8, 0.7254901960784313, 0.4549019607843137),
            (0.39215686274509803, 0.7098039215686275, 0.803921568627451),
            (0.6313725490196078, 0.788235294117647, 0.9568627450980393),
            (1.0, 0.7058823529411765, 0.5098039215686274),
            (0.5529411764705883, 0.8980392156862745, 0.6313725490196078),
            (1.0, 0.6235294117647059, 0.6078431372549019),
            (0.8156862745098039, 0.7333333333333333, 1.0),
            (0.8705882352941177, 0.7333333333333333, 0.6078431372549019),
            (0.9803921568627451, 0.6901960784313725, 0.8941176470588236),
            (0.8117647058823529, 0.8117647058823529, 0.8117647058823529),
            (1.0, 0.996078431372549, 0.6392156862745098),
            (0.7254901960784313, 0.9490196078431372, 0.9411764705882353)]

    if index is not None and shade is None:
        return get_color(color=color_order[index % len(colors)], shade= shades[index//len(colors)])
    
    elif index is not None:
        return get_color(color=color_order[index % len(colors)], shade=shade)

    elif (color is not None) and (shade is None):
        index = colors.index(color)
        return rgbs[index]

    elif (color is not None):
        index = shades.index(shade) * len(colors) + colors.index(color)
        return rgbs[index]

    elif (string is not None):
        if shade is None:
            shade_ = 'd'
        else:
            shade_ = shade
            
        if 'pca' in string:
            return get_color(color='aqua', shade=shade_)
        elif 'cpt' in string:
            return get_color(color='red', shade=shade_)
        elif 'flt' in string:
            return get_color(color='green', shade=shade_)
    
    else :
        index = shades.index(allinput[0]) * len(colors) + colors.index(allinput[1:])
        return rgbs[index]

def get_symbol(index=None,string=None):
    symbols = ['o','s','d','^','>','<','h','p','D','*']
    if index is not None:
        return symbols[index % len(symbols)]
    elif string is not None:
        if 'pca' in string:
            return get_symbol(0)
        elif 'cpt' in string:
            return get_symbol(1)
        elif 'flt' in string:
            return get_symbol(2)
        
def get_label(string):
    if 'pca' in string:
        return 'PCA+ANN'
    elif 'cpt' in string:
        return 'CMLM'
    elif 'flt' in string:
        return 'FGM+ANN'
    


onecol_figsize = (5.5,2.6)
onecol_figbounds = [0.16,0.2, 0.83, 0.75]

twocol_figsize = (4.4,3.15)
twocol_figbounds = [0.18, 0.17, 0.78, 0.81]

onecol_figsize_tall = (onecol_figsize[0],twocol_figsize[1])
onecol_figbounds_tall = [onecol_figbounds[0],
                         twocol_figbounds[1],
                         onecol_figbounds[2],
                         twocol_figbounds[3]]

          
