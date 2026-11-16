import nbformat, re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from PIL import Image as PI

ss=getSampleStyleSheet()
H1=ParagraphStyle('h1',parent=ss['Heading1'],textColor=colors.HexColor('#3b2fc9'),fontSize=15)
H2=ParagraphStyle('h2',parent=ss['Heading2'],fontSize=12,textColor=colors.HexColor('#222'))
B=ParagraphStyle('b',parent=ss['BodyText'],fontSize=9.5,leading=13)
BL=ParagraphStyle('bl',parent=B,leftIndent=14,bulletIndent=4)
def fmt(t):
    t=t.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
    t=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',t); t=re.sub(r'`(.+?)`',r'<font name="Courier">\1</font>',t)
    return t.replace('\u2192','->').replace('\u2194','<->').replace('\u2212','-').replace('\u00d7','x').replace('\u2248','~').replace('\u2265','>=').replace('\u2264','<=').replace('\u21d2','=>').replace('\u00a3','GBP ')
def img(p,w=17*cm):
    iw,ih=PI.open(p).size; return Image(p,width=w,height=w*ih/iw)

nb=nbformat.read('Task5_Titanic_EDA.ipynb',4)
obs={}  # figure -> observations; map by order of saved images
figs=['01_missing','02_histograms','03_countplots','04_survival_rates','05_boxplots','06_violin_kde','07_scatter','08_heatmap','09_pairplot','10_class_sex','11_family_cabin','12_outliers_skew']
titles=['Missing values','Histograms (age, fare, family size)','Count plots','Survival rate by category','Boxplots','Violin & KDE','Scatter plots','Correlation heatmap','Pairplot','Class x Sex x Survival','Family size & cabin','Outliers & skewness (fare)']
# collect observation markdown following each figure-producing code cell
cells=nb.cells; mp={}
for i,c in enumerate(cells):
    if c.cell_type=='code':
        for f in figs:
            if f"images/{f}.png" in c.source:
                nxt=cells[i+1]
                mp[f]=nxt.source if nxt.cell_type=='markdown' and nxt.source.startswith('**Observations') else ''
def bullets(src):
    out=[]
    for l in src.split('\n'):
        l=l.strip()
        if l.startswith('- '): out.append(Paragraph(fmt(l[2:]),BL,bulletText='\u2022'))
    return out

st=[]
st+=[Spacer(1,3*cm),Paragraph('Exploratory Data Analysis Report',ParagraphStyle('t',parent=ss['Title'],fontSize=26,textColor=colors.HexColor('#3b2fc9'))),
 Paragraph('Titanic Passenger Dataset',ParagraphStyle('s',parent=ss['Title'],fontSize=16)),Spacer(1,1*cm),
 Paragraph('Data Analyst Internship - Task 5',ParagraphStyle('c',parent=B,alignment=1,fontSize=12)),
 Paragraph('Tools: Python, Pandas, NumPy, Matplotlib, Seaborn',ParagraphStyle('c2',parent=B,alignment=1,fontSize=11)),Spacer(1,1*cm),
 Paragraph('<b>Objective:</b> Extract insights using visual and statistical exploration to find patterns, trends and anomalies in what influenced passenger survival.',B),PageBreak()]

# 1 overview
st+=[Paragraph('1. Dataset Overview',H1),
 Paragraph('The dataset holds <b>891 passengers</b> and <b>15 columns</b> (survived, pclass, sex, age, sibsp, parch, fare, embarked, class, who, adult_male, deck, embark_town, alive, alone). Only <b>38.4%</b> (342) survived.',B),Spacer(1,6)]
stats=[['Statistic','Age','Fare','SibSp','Parch'],['Count','714','891','891','891'],['Mean','29.70','32.20','0.52','0.38'],['Median','28.00','14.45','0','0'],['Std','14.53','49.69','1.10','0.81'],['Min','0.42','0.00','0','0'],['Max','80.00','512.33','8','6']]
t=Table(stats,hAlign='LEFT'); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#3b2fc9')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),.4,colors.grey),('FONTSIZE',(0,0),(-1,-1),9),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0eefc')])]))
st+=[Paragraph('Key numeric summary (<font name="Courier">.describe()</font>):',B),t,Spacer(1,8)]
st+=bullets("""- 577 males vs 314 females; 491 passengers (55%) travelled 3rd class; 644 (72%) embarked at Southampton.
- Mean fare (32.20) is more than double the median (14.45) -> strong right skew.
""")
st+=[Paragraph('2. Data Quality & Cleaning',H1)]
st+=bullets("""- `deck` is missing for 688 rows (77%) -> converted to a binary `has_cabin` flag.
- `age` missing for 177 rows (19.9%) -> imputed with median grouped by class and sex.
- `embarked` / `embark_town` missing 2 values -> filled with mode (Southampton).
- 107 duplicate-looking rows kept (no passenger ID; likely distinct passengers with identical attributes).
- Engineered features: `family_size`, `has_cabin`, `age_group`.""")
st+=[Spacer(1,6)]
for k in range(1):
    st+=[Paragraph(titles[0],H2),img(f'images/{figs[0]}.png',13*cm)]+bullets(mp[figs[0]])
st.append(PageBreak())

sect={1:'3. Univariate Analysis',3:'4. Bivariate Analysis',7:'5. Multivariate Analysis',11:'6. Outliers & Skewness'}
for i in range(1,12):
    f=figs[i]; blk=[]
    if i in sect and i!=11: st.append(Paragraph(sect[i],H1))
    if i==11: st.append(Paragraph(sect[11],H1))
    w=15*cm if f in('08_heatmap','09_pairplot') else 17*cm
    if f=='09_pairplot': w=13*cm
    st+=[KeepTogether([Paragraph(f'Figure {i+1}: {titles[i]}',H2),img(f'images/{f}.png',w)]),Paragraph('<b>Observations</b>',B)]+bullets(mp[f])+[Spacer(1,10)]
    if f in ('03_countplots','06_violin_kde','09_pairplot','10_class_sex'): st.append(PageBreak())
st.append(PageBreak())

st.append(Paragraph('7. Summary of Findings',H1))
rows=[['#','Finding','Evidence']]+[
['1','Overall survival 38.4%','342 / 891'],
['2','Sex is the strongest predictor','Women 74% vs men 19%'],
['3','Passenger class matters','63% / 47% / 24% (1st / 2nd / 3rd)'],
['4','Class x sex interaction','1st-class women 97% vs 3rd-class men 13.5%'],
['5','Children had an advantage','58% (age 0-12); seniors 23%'],
['6','Higher fare = higher survival','Median 26 (survived) vs 10.5 (died)'],
['7','Small families fared best','2-4 members: 55-72%; solo: 30%'],
['8','Cabin recorded = higher survival','67% vs 30%'],
['9','Embarkation effect is confounded by class','Cherbourg 55% vs Southampton 34%; 51% of Cherbourg was 1st class'],
['10','Data issues','deck 77% missing, age 20% missing, fare skew 4.8, pclass-fare r = -0.55, 15 zero fares']]
rows=[[Paragraph(fmt(c),B) for c in r] for r in rows]
t=Table(rows,colWidths=[1*cm,6.5*cm,9.5*cm],repeatRows=1); t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#3b2fc9')),('GRID',(0,0),(-1,-1),.4,colors.grey),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f0eefc')])]))
for c in t._cellvalues[0]: c.style=ParagraphStyle('w',parent=B,textColor=colors.white,fontName='Helvetica-Bold')
st+=[t,Spacer(1,10),Paragraph('<b>Conclusion:</b> Survival followed a "women and children first" pattern combined with a clear socio-economic gradient. Sex, passenger class, fare and family size are the key features for any predictive model. Fare should be log-transformed, age imputed by class/sex, and pclass/fare used carefully in linear models because of multicollinearity.',B),PageBreak()]

st.append(Paragraph('8. Interview Questions - Answers',H1))
qa=[c for c in cells if c.cell_type=='markdown' and c.source.startswith('## 9.')][0].source
for blk in qa.split('\n\n'):
    ls=[l for l in blk.split('\n') if not l.startswith('## ')]; st.append(Paragraph(fmt(ls[0]),B)); st.append(Paragraph(fmt(' '.join(ls[1:])),ParagraphStyle('a',parent=B,leftIndent=10,textColor=colors.HexColor('#333')))); st.append(Spacer(1,6))

def pn(c,d):
    c.saveState(); c.setFont('Helvetica',8); c.setFillColor(colors.grey); c.drawCentredString(A4[0]/2,1*cm,f'Titanic EDA Report  |  Page {d.page}'); c.restoreState()
SimpleDocTemplate('Task5_EDA_Report.pdf',pagesize=A4,leftMargin=2*cm,rightMargin=2*cm,topMargin=1.8*cm,bottomMargin=1.8*cm,title='Titanic EDA Report').build(st,onFirstPage=lambda c,d:None,onLaterPages=pn)
