using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.Diagnostics;

using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using System.Runtime.Serialization.Formatters.Binary;
using System.Runtime.InteropServices;

namespace VideoCropAnnotator{


	public partial class Form1 : Form{

		int currentFrameCount = -1;
		int currentFrameH = -1;
		int currentFrameW = -1;
		List<Mat> currentFrames;
		List<int> currentXAnnots;
		List<int> currentYAnnots;
		Rectangle cropBox = new Rectangle(100, 100, 100, 100);
		Image<Bgr, byte> doneStatusImg = new Image<Bgr, byte>(new Size(1, 1));

		bool hasLoadedFile = false;
		string loadedFile = "";

		public Form1(){
			InitializeComponent();
		}

		public void initDoneBar(){
			doneStatusImg = new Image<Bgr, byte>(new Size(currentFrameCount, 1));
		}
		public void setDoneBar(){
			byte[,,] data = doneStatusImg.Data;
			for (int j = currentFrameCount - 1; j >= 0; j--){
				if (currentXAnnots[j] > -1){
					data[0, j, 0] = 0;
					data[0, j, 1] = 255;
					data[0, j, 2] = 0;
				}else{
					data[0, j, 0] = 0;
					data[0, j, 1] = 0;
					data[0, j, 2] = 255;
				}
			}
			imgDone.Image = doneStatusImg;
			imgDone.Refresh();
		}


		public void updateGUI(bool active)
		{
			if (active)
			{
				hScrollBar1.Enabled = true;
				buttSave.Enabled = true;
				buttonReset.Enabled = true;
				buttonNext.Enabled = true;
				buttonPrev.Enabled = true;
				buttonPlay.Enabled = true;
				button1.Enabled = true;
				button2.Enabled = true;
				button3.Enabled = true;
				button4.Enabled = true;
				lblBox.Enabled = true;
				lblFrame.Enabled = true;
				lblMem.Enabled = true;
				imageBox1.Enabled = true;
				imageBox1.Height = currentFrameH;
				imageBox1.Width = currentFrameW;
				if (lblMode.Text.Length > 0)
				{
					string ctxt = lblMode.Text;
					int i1 = ctxt.IndexOf(":", StringComparison.Ordinal);
					int i2 = ctxt.IndexOf(" ", StringComparison.Ordinal);
					Double n1 = Convert.ToDouble(ctxt.Substring(0, i1));
					Double n2 = Convert.ToDouble(ctxt.Substring(i1 + 1, i2 - i1 - 1));
					int ch, cw;
					if (n1 <= n2)
					{
						ch = imageBox1.Height;
						cw = Convert.ToInt32((n1 / n2) * Convert.ToDouble(imageBox1.Height));
					}
					else
					{
						ch = Convert.ToInt32((n2 / n1) * Convert.ToDouble(imageBox1.Width));
						cw = imageBox1.Width;
					}
					cropBox.Height = ch;
					cropBox.Width = cw;
				}
				this.Size = new Size(Math.Max(currentFrameW + panel2.Width + 16, 600), Math.Max(currentFrameH + panel1.Height + 16, 480));
				hScrollBar1.Minimum = 1;
				hScrollBar1.Maximum = currentFrameCount;
				hScrollBar1.Value = 1;
				//hScrollBar1.Size = new Size(this.Size.Width+24, 46);
				//hScrollBar1.Location = new Point(-16, 46);
			}
			else
			{
				hScrollBar1.Enabled = false;
				buttSave.Enabled = false;
				buttonReset.Enabled = false;
				buttonNext.Enabled = false;
				buttonPrev.Enabled = false;
				buttonPlay.Enabled = false;
				button1.Enabled = false;
				button2.Enabled = false;
				button3.Enabled = false;
				button4.Enabled = false;
				lblBox.Text = "";
				lblDir.Text = "";
				lblMode.Text = "";
				lblFile.Text = "No file loaded";
				lblFrame.Text = "0/0";
				lblMem.Text = "0 MB";
				lblBox.Enabled = false;
				lblFrame.Enabled = false;
				lblMem.Enabled = false;
				hScrollBar1.Minimum = 1;
				hScrollBar1.Maximum = 1;
				hScrollBar1.Value = 1;
			}
			Application.DoEvents();
		}

		// load video
		public void loadVideo(string vidPath)
		{
			// reset info
			currentFrames = new List<Mat>();
			currentXAnnots = new List<int>();
			currentYAnnots = new List<int>();
			currentFrameCount = -1;
			currentFrameH = -1;
			currentFrameW = -1;

			// open video
			VideoCapture vidCap = new VideoCapture(vidPath);
			int currentFPS = Convert.ToInt32(vidCap.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.Fps));
			currentFrameCount = Convert.ToInt32(vidCap.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameCount));
			currentFrameH = Convert.ToInt32(vidCap.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight));
			currentFrameW = Convert.ToInt32(vidCap.GetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth));

			// read video
			currentFrameCount = 0;
			bool Reading = true;
			while (Reading)
			{
				Mat frame = vidCap.QueryFrame();
				if (frame != null)
				{
					currentFrameCount++;
					currentFrames.Add(frame.Clone());
					currentXAnnots.Add(-1);
					currentYAnnots.Add(-1);

					lblFrame.Text = "loading " + Convert.ToString(currentFrameCount);
					double ls = ((currentFrameCount * 3.0 * currentFrameW * currentFrameH * 2.0) / 1024.0) / 1024.0;
					lblMem.Text = Convert.ToString(Convert.ToInt32(ls)) + "MB";

					if (currentFrameCount % 50 == 0)
					{
						lblFrame.Refresh();
						lblMem.Refresh();
					}
				}
				else
				{
					Reading = false;
				}
			}

			// create done image
			initDoneBar();
			setDoneBar();

			// update GUI
			timer1.Interval = Convert.ToInt32(0.25 * (1000.0 / currentFPS));
			updateGUI(true);
			Application.DoEvents();
			show_frame();
			Application.DoEvents();
			imageBox1.Invalidate();
			Application.DoEvents();
			refresh_cropBox(Convert.ToInt32((imageBox1.Width / 2.0) - (cropBox.Width / 2.0)), Convert.ToInt32((imageBox1.Height / 2.0) - (cropBox.Height / 2.0)));
			Application.DoEvents();
		}

		// load video button
		private void button1_Click(object sender, EventArgs e) {
			hasLoadedFile = false;
			loadedFile = "";
			OpenFileDialog ofd = new OpenFileDialog();
			String vidPath;
			ofd.Title = "Select video file";
			ofd.Filter = "AVI Video file |*.avi";
			ofd.Multiselect = false;
			string root = Directory.GetCurrentDirectory();
			for (int i = 0; i < 6; i++) {
				Debug.WriteLine("Checking " + root + @"\DHF1k_video_selection");
				if (Directory.Exists(root + @"\DHF1k_video_selection"))
				{
					Debug.WriteLine(" Found!");
					ofd.InitialDirectory = root + @"\DHF1k_video_selection";
					break;
				}
				root = Directory.GetParent(root).ToString();
				if (i==5)
				{
					Debug.WriteLine(" Defaulting to C: ");
					ofd.InitialDirectory = @"C:\";
				}
			}


			// reset info
			currentFrames = new List<Mat>();
			currentXAnnots = new List<int>();
			currentYAnnots = new List<int>();
			currentFrameCount = -1;
			currentFrameH = -1;
			currentFrameW = -1;
			updateGUI(false);


			// show open file dialog
			if (ofd.ShowDialog() == DialogResult.OK){
				vidPath = ofd.FileName;

				// show file info
				lblFile.Text = "" + ofd.SafeFileName + "";
				lblDir.Text = "" + Path.GetDirectoryName(vidPath) + "";

				// select mode
				Form2 frm = new Form2();
				DialogResult qrwts = frm.ShowDialog();
				if (qrwts == DialogResult.Yes)
				{
					lblMode.Text = "1:3 (preserve height)";
					
				}
				else if (qrwts == DialogResult.No)
				{
					lblMode.Text = "3:1 (preserve width)";
				}
				else
				{
					updateGUI(false);
					return;
				}

				loadVideo(vidPath);
			}

		}

		// show frame
		public void show_frame(){
			lblFrame.Text = Convert.ToString(hScrollBar1.Value) + "/" + Convert.ToString(currentFrameCount);
			lblFrame.Refresh();
			imageBox1.Image = currentFrames[hScrollBar1.Value-1];
		}


		
		// save annotations
		private void buttSave_Click(object sender, EventArgs e){
			// check if all values are set
			bool annotEmpty = false;
			for (int i=0; i<currentFrameCount; i++)
			{
				if ( (currentXAnnots[i]==-1) || (currentYAnnots[i] == -1)){
					annotEmpty = true;
					break;
				}
			}
			if (annotEmpty)
			{
				DialogResult dv = MessageBox.Show("You haven't set a crop box for all video frames!\n\nAre you sure you want to continue and save an incomplete annotation file?", "Annotation Incomplete", MessageBoxButtons.YesNo);
				if (dv == DialogResult.No)
				{
					return;
				}
			}
			// create dir
			string annots_dir = Directory.GetCurrentDirectory() + @"\annotations_" + txtUser.Text + @"\";
			Directory.CreateDirectory(annots_dir);

			// create output file path string
			string fn;
			string fp;
			if (hasLoadedFile)
			{
				fn = loadedFile + "2";
				fp = loadedFile + "2";
			}
			else
			{
				fn = lblFile.Text.ToLower().Replace(".avi", "") + "." + lblMode.Text.Substring(0, 3).Replace(":", "-") + ".txt";
				fp = annots_dir + fn;
			}
			Debug.WriteLine("fn:" + fn);
			Debug.WriteLine("fp:" + fp);


			// Checking if scores.txt exists or not
			if (File.Exists(fp))
			{
				string lastModified = System.IO.File.GetLastWriteTime(fp).ToString();
				DialogResult dq = MessageBox.Show("Annotation file " + fp + " already exists!\n\nDate modified: " + lastModified + "\n\nAre you sure you want to overwrite it?", "Overwrite Existing Annotations", MessageBoxButtons.YesNo);
				if (dq == DialogResult.No)
				{
					return;
				}
			}
				
			// write to file
			using (FileStream fs = File.Create(fp)) 
			{
				StreamWriter sw = new StreamWriter(fs);
				for (int i =0; i< currentFrameCount; i++)
				{
					sw.WriteLine(currentXAnnots[i].ToString() + "," + currentYAnnots[i].ToString());
				}
				sw.Close();
				Application.DoEvents();
			}

			// message box for success info
			DialogResult ds = MessageBox.Show("Successfully wrote "+ currentFrameCount.ToString() + " at " + fp, "Success", MessageBoxButtons.OK);
		}


		private void hScrollBar1_Scroll(object sender, ScrollEventArgs e){
			show_frame();
		}

		private void hScrollBar1_ValueChanged(object sender, EventArgs e){
			show_frame();
		}



		bool drawing = false;

		public void refresh_cropBox(int x, int y){
			if (currentFrameCount == -1) return;

			string ctxt = lblMode.Text;
			int i1 = ctxt.IndexOf(":", StringComparison.Ordinal);
			int i2 = ctxt.IndexOf(" ", StringComparison.Ordinal);
			Double n1 = Convert.ToDouble(ctxt.Substring(0, i1));
			Double n2 = Convert.ToDouble(ctxt.Substring(i1 + 1, i2 - i1 - 1));

			int ch, cw;
			if (n1<=n2){
				ch = imageBox1.Height;
				cw = Convert.ToInt32((n1/n2) * Convert.ToDouble(imageBox1.Height));
			}
			else{
				ch = Convert.ToInt32((n2/n1) * Convert.ToDouble(imageBox1.Width));
				cw = imageBox1.Width;
			}
			cropBox.Height = ch;
			cropBox.Width = cw;

			if (currentXAnnots[hScrollBar1.Value-1] == -1)
			{
				int cx = Math.Min(Math.Max(0, x - Convert.ToInt32(cropBox.Width / 2)), imageBox1.Width - cropBox.Width);
				int cy = Math.Min(Math.Max(0, y - Convert.ToInt32(cropBox.Height / 2)), imageBox1.Height - cropBox.Height);
				currentXAnnots[hScrollBar1.Value-1] = cx;
				currentYAnnots[hScrollBar1.Value-1] = cy;
			}


			lblBox.Text = "w:" + Convert.ToString(imageBox1.Width) + ",h:" + Convert.ToString(imageBox1.Height) + "\ncx:" + Convert.ToString(currentXAnnots[hScrollBar1.Value-1]) + "," + "cy:" + Convert.ToString(currentYAnnots[hScrollBar1.Value-1]) + ",cw:" + Convert.ToString(cw) + ",ch:" + Convert.ToString(ch);
		}



		private void Form1_Load(object sender, EventArgs e){
			try
			{
				string userName = System.Security.Principal.WindowsIdentity.GetCurrent().Name;
				int index = userName.LastIndexOf(@"\");
				userName = userName.Substring(index + 1);
				txtUser.Text = userName;
			}
			catch (Exception err)
			{
				Debug.WriteLine(err.ToString());
			}
		}

		private void buttonReset_Click(object sender, EventArgs e){
			DialogResult dialogResult = MessageBox.Show("This will reset annotatios. Are you sure?", "Reset Annotations", MessageBoxButtons.YesNo);
			if (dialogResult == DialogResult.Yes){
				for (int i=0; i<currentFrameCount; i++){
					currentXAnnots[i] = -1;
					currentYAnnots[i] = -1;
				}

				initDoneBar();
				setDoneBar();
				
				updateGUI(true);
				Application.DoEvents();
				show_frame();
				Application.DoEvents();
				imageBox1.Invalidate();
				Application.DoEvents();
			}
			else if (dialogResult == DialogResult.No)	{
				//do something else
			}
		}

		private void buttonPlay_Click(object sender, EventArgs e){
			if (timer1.Enabled)	{
				buttonPlay.Text = "▶";
				buttonNext.Enabled = true;
				buttonPrev.Enabled = true;
				timer1.Enabled = false;
			}else{
				buttonPlay.Text = "⏸";
				buttonNext.Enabled = false;
				buttonPrev.Enabled = false;
				timer1.Enabled = true;
			}
		}

		private void timer1_Tick(object sender, EventArgs e){
			if (hScrollBar1.Value  == hScrollBar1.Maximum)
			{
				buttonPlay.Text = "▶";
				buttonNext.Enabled = true;
				buttonPrev.Enabled = true;
				timer1.Enabled = false;
				return;
			}
			if (hScrollBar1.Value + 1 >= hScrollBar1.Maximum){
				buttonPlay.Text = "▶";
				buttonNext.Enabled = true;
				buttonPrev.Enabled = true;
				timer1.Enabled = false;
			}
			hScrollBar1.Value += 1;
			show_frame();
			if (hScrollBar1.Value == hScrollBar1.Maximum){
				timer1.Enabled = false;
			}

		}

		private void buttonNext_Click(object sender, EventArgs e){
			hScrollBar1.Value = Math.Min(hScrollBar1.Value + 10, hScrollBar1.Maximum);
			show_frame();
		}
		private void button2_Click(object sender, EventArgs e)
		{
			hScrollBar1.Value = Math.Min(hScrollBar1.Value + 1, hScrollBar1.Maximum);
			show_frame();
		}
		private void buttonPrev_Click(object sender, EventArgs e){
			hScrollBar1.Value = Math.Max(hScrollBar1.Value - 10, hScrollBar1.Minimum);
			show_frame();
		}
		private void button1_Click_1(object sender, EventArgs e)
		{
			hScrollBar1.Value = Math.Max(hScrollBar1.Value - 1, hScrollBar1.Minimum);
			show_frame();
		}


		private void hScrollBar1_Scroll_1(object sender, ScrollEventArgs e){
			show_frame();
		}





		int lastIndexOfEmpty = -1;


		private void imageBox1_MouseDown(object sender, MouseEventArgs e)
		{
			if ((e.Button == MouseButtons.Left) & (timer1.Enabled == false))
			{
				drawing = true;
			}
		}

		private void imageBox1_MouseUp(object sender, MouseEventArgs e)
		{
			if ((e.Button == MouseButtons.Left) & (timer1.Enabled == false))
			{
				drawing = false;

				// find previous last index that is set
				if (hScrollBar1.Value == 1)
				{
					lastIndexOfEmpty = 0;
				}
				else
				{
					for (int i = hScrollBar1.Value - 1 - 1; i >= 0; i--)
					{
						if (currentXAnnots[i] == -1)
							continue;
						lastIndexOfEmpty = i;
						break;
					}
				}

				// debug print
				Debug.WriteLine("lastIndexOfEmpty:" + lastIndexOfEmpty.ToString());
				Debug.WriteLine("hScrollBar1.Value:" + hScrollBar1.Value.ToString());

				// compute step
				int cX = currentXAnnots[hScrollBar1.Value - 1];
				int cY = currentYAnnots[hScrollBar1.Value - 1];
				int sX = currentXAnnots[lastIndexOfEmpty];
				int sY = currentYAnnots[lastIndexOfEmpty];
				int rangeX = cX - sX;
				int rangeY = cY - sY;
				double steps = Convert.ToDouble(hScrollBar1.Value - lastIndexOfEmpty);

				// debug print
				Debug.WriteLine("steps:" + steps.ToString());

				// interpolate
				if (steps > 0)
				{
					double stepX = rangeX / steps;
					double stepY = rangeY / steps;
					int c = -1;
					for (int i = lastIndexOfEmpty; i < hScrollBar1.Value - 1; i++)
					{
						c++;
						currentXAnnots[i] = Convert.ToInt32(sX + (stepX * c));
						currentYAnnots[i] = Convert.ToInt32(sY + (stepY * c));
					}
				}

				setDoneBar();
			}
		}

		private void imageBox1_MouseMove(object sender, MouseEventArgs e)
		{
			if (drawing)
			{
				if (e.Button == MouseButtons.Left)
				{
					int cX = Math.Min(Math.Max(0, e.X - Convert.ToInt32(cropBox.Width / 2)), imageBox1.Width - cropBox.Width);
					int cY = Math.Min(Math.Max(0, e.Y - Convert.ToInt32(cropBox.Height / 2)), imageBox1.Height - cropBox.Height);
					currentXAnnots[hScrollBar1.Value - 1] = cX;
					currentYAnnots[hScrollBar1.Value - 1] = cY;
					refresh_cropBox(e.X, e.Y);
					imageBox1.Invalidate();
				}
			}
		}

		private void imageBox1_Paint(object sender, PaintEventArgs e)
		{
			if (currentFrameCount == -1) return;

			Brush brush;
			Pen pen;
			int X = currentXAnnots[hScrollBar1.Value - 1];
			int Y = currentYAnnots[hScrollBar1.Value - 1];
			if (X == -1)
			{
				X = Convert.ToInt32((imageBox1.Width / 2.0) - (cropBox.Width / 2.0));
				Y = Convert.ToInt32((imageBox1.Height / 2.0) - (cropBox.Height / 2.0));
				brush = new SolidBrush(Color.FromArgb(20, 255, 0, 50));
				pen = new Pen(Color.FromArgb(255, 0, 50));
			}
			else
			{
				brush = new SolidBrush(Color.FromArgb(20, 0, 255, 50));
				pen = new Pen(Color.FromArgb(0, 255, 50));
			}
			Rectangle cropBox2 = new Rectangle(X, Y, cropBox.Width - 4, cropBox.Height - 4);
			e.Graphics.FillRectangle(brush, cropBox2);
			e.Graphics.DrawRectangle(pen, cropBox2);
		}

		private void imageBox1_Click(object sender, EventArgs e)
		{

		}

		private void button4_Click(object sender, EventArgs e)
		{
			hScrollBar1.Value = Math.Max(hScrollBar1.Value - 20, hScrollBar1.Minimum);
			show_frame();
		}

		private void button3_Click(object sender, EventArgs e)
		{
			hScrollBar1.Value = Math.Min(hScrollBar1.Value + 20, hScrollBar1.Maximum);
			show_frame();
		}

		OpenFileDialog ofd_load = new OpenFileDialog();
		private void button5_Click(object sender, EventArgs e)
		{
			ofd_load.Title = "Select annotations file";
			ofd_load.Filter = "Text file |*.txt";
			ofd_load.Multiselect = false;


			if (ofd_load.ShowDialog() == DialogResult.OK)
			{
				string txtPath = ofd_load.FileName;
				int pos_end;
				if (txtPath.Contains("1-3")) 
				{
					lblMode.Text = "1:3 (preserve height)";
					pos_end = txtPath.IndexOf(".1-3.txt");
				}
				else
				{
					pos_end = txtPath.IndexOf(".3-1.txt");
					lblMode.Text = "3:1 (preserve width)";
				}
				int pos_start = txtPath.IndexOf(".")+1;

				Debug.WriteLine(txtPath);
				Debug.WriteLine(pos_start);
				Debug.WriteLine(pos_end);
				string vid_no = txtPath.Substring(pos_start, pos_end - pos_start);
				string root_path = "D:\\smart_crop_annotations\\DHF1k_video_selection\\";
				string vid_path = root_path + vid_no + ".avi";
				Debug.WriteLine(vid_path);
				loadVideo(vid_path);

				hasLoadedFile = true;
				loadedFile = txtPath;
				int counter = 0;
				string line;
				System.IO.StreamReader file = new System.IO.StreamReader(txtPath);
				while ((line = file.ReadLine()) != null)
				{
					int x=-1;
					int y=-1;
					try
					{ 
						x = Convert.ToInt32(line.Split(",")[0]);
						y = Convert.ToInt32(line.Split(",")[1]);
					}
					catch (Exception)
					{
						break;
					}

					currentXAnnots[counter] = x;
					currentYAnnots[counter] = y;
					counter++;
				}

				initDoneBar();
				setDoneBar();
			}


		}
	}
}

