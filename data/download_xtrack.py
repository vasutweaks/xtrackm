import ftplib
import os

avuser = "srinivasu.u@incois.gov.in"
avpass = "B9FQgr"

ftpsite = "ftp-access.aviso.altimetry.fr"
print(dir(ftplib))

zone = "NINDIAN"
version = "v2.2"
path_old = f"/regional-xtrack-coastal/version_xtrack_l2p_2022"
path_new = f"/regional-xtrack-coastal/version_x_track_l2p_2022"
sats = ["GFO", "TP+J1+J2+J3", "S3A", "TPN+J1N+J2N", "ERS1+ERS2+ENV+SRL", "HY2"]

sats_new = ["GFO", "TP+J1+J2+J3+S6A", "S3A", "S3B", "TPN+J1N+J2N+J3N", "ERS1+ERS2+ENV+SRL", "HY2A", "HY2B"]
# sats_new = ["TP+J1+J2+J3+S6A", "S3A", "S3B", "TPN+J1N+J2N+J3N", "ERS1+ERS2+ENV+SRL", "HY2A", "HY2B"]

ftp = ftplib.FTP(ftpsite)
ftp.login(avuser, avpass)
print(ftp.getwelcome())
for sat in sats_new:
    # if sat directory does not exist create it
    if not os.path.exists(f"{sat}"):
        os.makedirs(f"{sat}")
    cddir = (
        f"{path_new}/{version}/{zone}/{sat}/SLA"
    )
    ftp.cwd(cddir)
    # ftp.retrlines('LIST')
    flist = ftp.nlst()
    for f in flist:
        print(f)
        f_out = f"{sat}/{f}"
        # if not os.path.exists(f_out) and f.endswith(".lzma"):
        if not os.path.exists(f_out):
            print(f_out, "is not here")
            with open(f_out, "wb") as fh:
                ftp.retrbinary("RETR " + f, fh.write)

ftp.quit()

# drwxr-xr-x   3 10845    100          1035 May 31 14:01 GFO
# drwxr-xr-x   3 10845    100          1035 May 31 14:02 TP+J1+J2+J3
# drwxr-xr-x   3 10845    100          1035 May 31 14:02 S3A
# drwxr-xr-x   3 10845    100          1035 May 31 14:02 TPN+J1N+J2N
# drwxr-xr-x   3 10845    100          1035 May 31 14:02 HY2
# drwxr-xr-x   3 10845    100          1035 Jul  3 12:30 ERS1+ERS2+ENV+SRL
#
