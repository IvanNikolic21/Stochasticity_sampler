"""Contains FirstLight processing to get some high-z scaling relations."""

import numpy as np
import json

def SFRvsMh(z, sample_Ms = True):
    ff=open('/home/inikolic/projects/stochasticity/FirstLight_database.dat', 'r')
    database = json.load(ff)
    ff.close()
    extract_keys_base=[ 'FL633', 'FL634', 'FL635', 'FL637', 'FL638', 'FL639', 'FL640', 'FL641', 'FL642', 'FL643', 'FL644', 'FL645', 'FL646',
                       'FL647', 'FL648', 'FL649', 'FL650', 'FL651', 'FL653', 'FL655', 'FL656', 'FL657', 'FL658', 'FL659', 'FL660', 'FL661',
                       'FL662', 'FL664', 'FL665', 'FL666', 'FL667', 'FL668', 'FL669', 'FL670', 'FL671', 'FL673', 'FL675', 'FL676', 'FL678',
                       'FL679', 'FL680', 'FL681', 'FL682', 'FL683', 'FL684', 'FL685', 'FL686', 'FL687', 'FL688', 'FL689', 'FL690', 'FL691', 
                       'FL692', 'FL693', 'FL694', 'FL695', 'FL696', 'FL697', 'FL698', 'FL700', 'FL701', 'FL702', 'FL703', 'FL704', 'FL705', 
                       'FL706', 'FL707', 'FL708', 'FL709', 'FL710', 'FL711', 'FL712', 'FL713', 'FL714', 'FL715', 'FL716', 'FL717', 'FL718',
                       'FL719', 'FL720', 'FL721', 'FL722', 'FL723', 'FL724', 'FL725', 'FL726', 'FL727', 'FL728', 'FL729', 'FL730', 'FL731',
                       'FL732', 'FL733', 'FL734', 'FL735', 'FL736', 'FL739', 'FL740', 'FL744', 'FL745', 'FL746', 'FL747', 'FL748', 'FL750',
                       'FL752', 'FL753', 'FL754', 'FL755', 'FL756', 'FL757', 'FL758', 'FL760', 'FL761', 'FL763', 'FL767', 'FL768', 'FL769',
                       'FL770', 'FL771', 'FL772', 'FL773', 'FL774', 'FL775', 'FL777', 'FL778', 'FL779', 'FL780', 'FL781', 'FL782', 'FL783',
                       'FL784', 'FL788', 'FL789', 'FL792', 'FL793', 'FL794', 'FL796', 'FL800', 'FL801', 'FL802', 'FL803', 'FL805', 'FL806',
                       'FL808', 'FL809', 'FL810', 'FL811', 'FL814', 'FL815', 'FL816', 'FL818', 'FL819', 'FL823', 'FL824', 'FL825', 'FL826',
                       'FL827', 'FL829',         'FL652', 'FL654', 'FL699', 'FL738', 'FL741', 'FL742', 'FL749', 'FL762', 'FL764', 'FL766',
                       'FL776', 'FL785', 'FL786', 'FL787', 'FL790', 'FL795', 'FL797', 'FL798', 'FL799', 'FL804', 'FL807', 'FL812', 'FL813',
                       'FL817', 'FL820', 'FL822', 'FL828', 'FL839', 'FL850', 'FL855', 'FL862', 'FL870', 'FL874', 'FL879', 'FL882', 'FL891',
                       'FL896', 'FL898', 'FL903', 'FL904', 'FL906', 'FL915', 'FL916', 'FL917', 'FL918', 'FL919', 'FL921', 'FL922', 'FL923',
                       'FL924', 'FL925', 'FL926', 'FL928', 'FL930', 'FL932', 'FL934', 'FL935', 'FL765', 'FL835', 'FL836', 'FL841', 'FL843',
                       'FL844', 'FL845', 'FL846', 'FL847', 'FL848', 'FL849', 'FL851', 'FL856', 'FL857', 'FL858', 'FL859', 'FL860', 'FL861',
                       'FL865', 'FL866', 'FL867', 'FL868', 'FL871', 'FL877', 'FL881', 'FL884', 'FL885', 'FL887', 'FL888', 'FL893', 'FL895',
                       'FL897', 'FL899', 'FL900', 'FL901', 'FL907', 'FL908', 'FL909', 'FL911', 'FL912', 'FL837', 'FL838', 'FL840', 'FL852',
                       'FL853', 'FL863', 'FL864', 'FL869', 'FL872', 'FL873', 'FL883', 'FL890', 'FL894', 'FL902', 'FL834', 'FL854', 'FL876',
                       'FL913', 'FL927', 'FL931', 'FL933', 'FL938', 'FL939', 'FL940', 'FL942', 'FL943', 'FL946', 'FL947', 'FL914', 'FL920',
                       'FL929', 'FL936', 'FL937', 'FL941', 'FL944' ]
    extract_a='a'
    extract_run='run'
    extract_field1='Mvir'
    extract_field2='Ms'
    extract_field3='SFR'
    s='_'
    possible_a = np.array([0.059, 0.060, 0.062, 0.064, 0.067, 0.069, 0.071, 0.074, 0.077, 0.080, 0.083, 0.087, 0.090, 0.095, 0.1, 0.105, 0.111, 0.118, 0.125,
                 0.133, 0.142, 0.154, 0.160])
    possible_a_str = list(map("{:.3f}".format, possible_a))
    z_poss = 1./possible_a - 1
    z_actual = min(z_poss, key=lambda x:abs(x-z))
    #print(possible_a_str,)
    index_actual = np.argmin(abs(np.array(z_poss)-z))
    #print(possible_a_str[index_actual])
    Ms = []
    Mh = []
    SFR = []
    for j in range(len(possible_a_str)):
        extract_a_value = possible_a_str[index_actual]
        for i in range(len(extract_keys_base)):
            extract_keys_run= extract_keys_base[i] + s + extract_a_value + s + extract_run
            extract_keys_a= extract_keys_base[i] + s + extract_a_value + s + extract_a
            extract_keys_field1= extract_keys_base[i] + s + extract_a_value + s + extract_field1
            extract_keys_field2= extract_keys_base[i] + s + extract_a_value + s + extract_field2
            extract_keys_field3= extract_keys_base[i] + s + extract_a_value + s + extract_field3
            if ((extract_keys_field1) in database) and ((extract_keys_field2) in database):
                #print(database[extract_keys_a], database[extract_keys_run], "%.2e" % database[extract_keys_field1],
                #      "%.2e" % database[extract_keys_field2], database[extract_keys_field3])
                Mh.append(database[extract_keys_field1])
                Ms.append(database[extract_keys_field2])
                SFR.append(database[extract_keys_field3])
#    return Mh, Ms, SFR
    Mh_uniq=set()
    Ms_uniq=set()
    SFR_uniq=set()
    Mh_uniq_list = []
    SFR_uniq_list = []
    Ms_uniq_list = []
    for i, mh in enumerate(Mh):
        if (mh, SFR[i], Ms[i]) not in Mh_uniq:
            Mh_uniq.add((mh, SFR[i], Ms[i]))
            Mh_uniq_list.append(mh)
            SFR_uniq_list.append(SFR[i])
            Ms_uniq_list.append(Ms[i])
    zipped = zip(Mh_uniq_list,SFR_uniq_list, Ms_uniq_list)
    zipped = sorted(zipped)
    Mh_uniq_sorted, SFR_uniq_sorted, Ms_uniq_sorted = zip(*zipped)
#    print(Mh_uniq_sorted, SFR_uniq_sorted)
    if sample_Ms:
        a_ms, b_ms = np.polyfit (np.log10(Mh_uniq_sorted), np.log10(Ms_uniq_sorted), 1 ) 
        bins = 2
        binned_masses=np.linspace(np.log10(min(Mh_uniq_sorted)),np.log10(max(Mh_uniq_sorted)), bins)
        means_ms=np.zeros(bins-1)
        number_of_masses=np.zeros(bins-1)
        errs_ms = np.zeros(bins-1)

        for i, mass in enumerate(Mh_uniq_sorted):
            for j, bining in enumerate(binned_masses):
                if j<(bins-2) and (np.log10(mass)>=bining) and (np.log10(mass)<binned_masses[j+1]):
                    err_ms[j]+=(((a_ms*np.log10(mass)+b_ms) - np.log10(Ms_uniq_sorted[i]))**2)
                    number_of_masses[j]+=1
                    means_ms[j]+=(mass)
                    break
                elif(j==bins-2):
                    errs_ms[bins-2]+=(((a_ms*np.log10(mass)+b_ms) - np.log10(Ms_uniq_sorted[i]))**2)
                    number_of_masses[bins-2]+=1
                    means_ms[bins-2]+=(mass)
                    break
        errs_ms=np.sqrt(errs_ms/number_of_masses)
        means_ms=np.sqrt(means_ms/number_of_masses)
    
        #do the same for Ms-SFR
    
        a_sfr, b_sfr = np.polyfit (np.log10(Ms_uniq_sorted), np.log10(SFR_uniq_sorted), 1 )
        binned_stellar_masses=np.linspace(np.log10(min(Ms_uniq_sorted)),np.log10(max(Ms_uniq_sorted)), bins)
        means_sfr=np.zeros(bins-1)
        errs_sfr = np.zeros(bins-1)
        number_of_stellar_masses = np.zeros(bins-1)
        for i, stellar_mass in enumerate(Ms_uniq_sorted):
            for j, stellar_bining in enumerate(binned_stellar_masses):
                if j<(bins-2) and (np.log10(stellar_mass)>=stellar_bining) and (np.log10(stellar_mass)<binned_stellar_masses[j+1]):
                    err_sfr[j]+=(((a_sfr*np.log10(stellar_mass)+b_sfr) - np.log10(SFR_uniq_sorted[i]))**2)
                    number_of_stellar_masses[j]+=1
                    means_sfr[j]+=(stellar_mass)
                    break
                elif(j==bins-2):
                    errs_sfr[bins-2]+=(((a_sfr*np.log10(stellar_mass)+b_sfr) - np.log10(SFR_uniq_sorted[i]))**2)
                    number_of_stellar_masses[bins-2]+=1
                    means_sfr[bins-2]+=(stellar_mass)
                    break
        errs_sfr = np.sqrt(errs_sfr/number_of_stellar_masses)
        means_sfr = np.sqrt(means_sfr/number_of_stellar_masses)
        return (means_ms, errs_ms, a_ms,b_ms, means_sfr, errs_sfr, a_sfr, b_sfr)

    else:
        a_ms, b_ms = np.polyfit (np.log10(Mh_uniq_sorted), np.log10(SFR_uniq_sorted), 1 ) 
        bins = 2
        binned_masses=np.linspace(np.log10(min(Mh_uniq_sorted)),np.log10(max(Mh_uniq_sorted)), bins)
        means_ms=np.zeros(bins-1)
        number_of_masses=np.zeros(bins-1)
        errs_ms = np.zeros(bins-1)

        for i, mass in enumerate(Mh_uniq_sorted):
            for j, bining in enumerate(binned_masses):
                if j<(bins-2) and (np.log10(mass)>=bining) and (np.log10(mass)<binned_masses[j+1]):
                    err_ms[j]+=(((a_ms*np.log10(mass)+b_ms) - np.log10(SFR_uniq_sorted[i]))**2)
                    number_of_masses[j]+=1
                    means_ms[j]+=(mass)
                    break
                elif(j==bins-2):
                    errs_ms[bins-2]+=(((a_ms*np.log10(mass)+b_ms) - np.log10(SFR_uniq_sorted[i]))**2)
                    number_of_masses[bins-2]+=1
                    means_ms[bins-2]+=(mass)
                    break
        errs_ms=np.sqrt(errs_ms/number_of_masses)
        means_ms=np.sqrt(means_ms/number_of_masses)
        
        return (means_ms, errs_ms, a_ms,b_ms)

def SFRvsMh_lin(z, sample_Ms = True, z_min=7, z_max=15, bins=20, pol = 1):
    """
        Returns the SFRvsMh or SFRvsMs and MsvsMh relation parameters, but with a liner evolution of the parameters with redshift.
        Parameters
        ----------
        z: float,
            redshift at which the relations are taken.
        sample_Ms: boolean; optional,
            if True, SFRvsMs and MsvsMh relations are given, if not SFRvsMh is given. Default is True.
        z_min: float; optional,
            minimal redshift for which the liner fit it taken. Default is 7.0.
        z_max: float; optional,
            maximum redshift for which the linear fit is taken. Default is 15.0.
        bins: int, optional,
            number of samples taken from the database. Default is 20.
        pol: int, optional,
            order of the polynomial for the fit. Default is 1, i.e. linear fit. 
            The order of the fit is the same for both relations if sample_Ms is True.
        returns
        ----------
        a_SFR: float,
            linear part of the SRF/Mh relation. If sample Ms is True, then coefficient of SFR/Ms.
        b_SFR: float,
            b coefficient of the SFR/Mh relation. If sample Ms is True, then coefficient of SFR/Ms.
        sigma_SFR: float,
            standard deviation of the SFR/Mh relation. If sample Ms is True, then deviation of SFR/Ms.
        a_Ms: float,
            if sample_Ms is True. linear coefficient of Ms/Mh relation.
        b_Ms: float,
            if sample_Ms is True. b coefficient of Ms/Mh relation.
        sigma_Ms: float,
            if sample_Ms is True. Standard deviation of the Ms/Mh relation.
    """
    
    a_SFR_arr = np.zeros(shape=bins)
    b_SFR_arr = np.zeros(shape=bins)
    sigma_SFR_arr = np.zeros(shape=bins)
    if sample_Ms:
        a_Ms_arr = np.zeros(shape=bins)
        b_Ms_arr = np.zeros(shape=bins)
        sigma_Ms_arr = np.zeros(shape=bins)
    redshifts = np.linspace(z_min, z_max, bins)
    
    for index, z_sam in enumerate(redshifts):
        if sample_Ms:
            _, sigma_Ms_arr[index], a_Ms_arr[index], b_Ms_arr[index], _, sigma_SFR_arr[index], a_SFR_arr[index], b_SFR_arr[index] = SFRvsMh(z=z_sam, sample_Ms = sample_Ms)
        else: 
            _, sigma_SFR_arr[index], a_SFR_arr[index], b_SFR_arr[index] = SFRvsMh(z=z_sam, sample_Ms = sample_Ms)
    
    a_SFR_lin, a_SFR_coef = np.polyfit(redshifts, a_SFR_arr, pol )
    b_SFR_lin, b_SFR_coef = np.polyfit(redshifts, b_SFR_arr, pol )
    sigma_SFR_lin, sigma_SFR_coef = np.polyfit(redshifts, sigma_SFR_arr, pol )
    if sample_Ms:
        a_Ms_lin, a_Ms_coef = np.polyfit(redshifts, a_Ms_arr, pol )
        b_Ms_lin, b_Ms_coef = np.polyfit(redshifts, b_Ms_arr, pol )
        sigma_Ms_lin, sigma_Ms_coef = np.polyfit(redshifts, sigma_Ms_arr, pol )
    
    if sample_Ms:
        return a_SFR_lin * z + a_SFR_coef, b_SFR_lin * z + b_SFR_coef, sigma_SFR_lin * z + sigma_SFR_coef,\
               a_Ms_lin * z + a_Ms_coef, b_Ms_lin * z + b_Ms_coef, sigma_Ms_lin * z + sigma_Ms_coef
    else:
        return a_SFR_lin * z + a_SFR_coef, b_SFR_lin * z + b_SFR_coef, sigma_SFR_lin * z + sigma_SFR_coef
