
namespace VideoCropAnnotator
{
	partial class Form1
	{
		/// <summary>
		///  Required designer variable.
		/// </summary>
		private System.ComponentModel.IContainer components = null;

		/// <summary>
		///  Clean up any resources being used.
		/// </summary>
		/// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
		protected override void Dispose(bool disposing)
		{
			if (disposing && (components != null))
			{
				components.Dispose();
			}
			base.Dispose(disposing);
		}

		#region Windows Form Designer generated code

		/// <summary>
		///  Required method for Designer support - do not modify
		///  the contents of this method with the code editor.
		/// </summary>
		private void InitializeComponent()
		{
			this.components = new System.ComponentModel.Container();
			System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
			this.buttonLoad = new System.Windows.Forms.Button();
			this.buttSave = new System.Windows.Forms.Button();
			this.imageList1 = new System.Windows.Forms.ImageList(this.components);
			this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
			this.label1 = new System.Windows.Forms.Label();
			this.label2 = new System.Windows.Forms.Label();
			this.label4 = new System.Windows.Forms.Label();
			this.lblMem = new System.Windows.Forms.Label();
			this.imageBox1 = new Emgu.CV.UI.ImageBox();
			this.label5 = new System.Windows.Forms.Label();
			this.lblBox = new System.Windows.Forms.Label();
			this.label6 = new System.Windows.Forms.Label();
			this.buttonReset = new System.Windows.Forms.Button();
			this.panel1 = new System.Windows.Forms.Panel();
			this.button4 = new System.Windows.Forms.Button();
			this.button3 = new System.Windows.Forms.Button();
			this.button2 = new System.Windows.Forms.Button();
			this.button1 = new System.Windows.Forms.Button();
			this.lblFrame = new System.Windows.Forms.TextBox();
			this.buttonPrev = new System.Windows.Forms.Button();
			this.buttonNext = new System.Windows.Forms.Button();
			this.buttonPlay = new System.Windows.Forms.Button();
			this.imgDone = new Emgu.CV.UI.ImageBox();
			this.hScrollBar1 = new System.Windows.Forms.HScrollBar();
			this.panel2 = new System.Windows.Forms.Panel();
			this.button5 = new System.Windows.Forms.Button();
			this.lblFile = new System.Windows.Forms.TextBox();
			this.lblDir = new System.Windows.Forms.TextBox();
			this.lblMode = new System.Windows.Forms.Label();
			this.txtUser = new System.Windows.Forms.TextBox();
			this.label7 = new System.Windows.Forms.Label();
			this.timer1 = new System.Windows.Forms.Timer(this.components);
			((System.ComponentModel.ISupportInitialize)(this.imageBox1)).BeginInit();
			this.panel1.SuspendLayout();
			((System.ComponentModel.ISupportInitialize)(this.imgDone)).BeginInit();
			this.panel2.SuspendLayout();
			this.SuspendLayout();
			// 
			// buttonLoad
			// 
			resources.ApplyResources(this.buttonLoad, "buttonLoad");
			this.buttonLoad.Name = "buttonLoad";
			this.buttonLoad.UseVisualStyleBackColor = true;
			this.buttonLoad.Click += new System.EventHandler(this.button1_Click);
			// 
			// buttSave
			// 
			resources.ApplyResources(this.buttSave, "buttSave");
			this.buttSave.Name = "buttSave";
			this.buttSave.UseVisualStyleBackColor = true;
			this.buttSave.Click += new System.EventHandler(this.buttSave_Click);
			// 
			// imageList1
			// 
			this.imageList1.ColorDepth = System.Windows.Forms.ColorDepth.Depth8Bit;
			resources.ApplyResources(this.imageList1, "imageList1");
			this.imageList1.TransparentColor = System.Drawing.Color.Transparent;
			// 
			// openFileDialog1
			// 
			this.openFileDialog1.FileName = "openFileDialog1";
			// 
			// label1
			// 
			resources.ApplyResources(this.label1, "label1");
			this.label1.Name = "label1";
			// 
			// label2
			// 
			resources.ApplyResources(this.label2, "label2");
			this.label2.Name = "label2";
			// 
			// label4
			// 
			resources.ApplyResources(this.label4, "label4");
			this.label4.Name = "label4";
			// 
			// lblMem
			// 
			this.lblMem.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
			resources.ApplyResources(this.lblMem, "lblMem");
			this.lblMem.Name = "lblMem";
			// 
			// imageBox1
			// 
			this.imageBox1.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
			this.imageBox1.FunctionalMode = Emgu.CV.UI.ImageBox.FunctionalModeOption.Minimum;
			resources.ApplyResources(this.imageBox1, "imageBox1");
			this.imageBox1.Name = "imageBox1";
			this.imageBox1.TabStop = false;
			this.imageBox1.Click += new System.EventHandler(this.imageBox1_Click);
			this.imageBox1.Paint += new System.Windows.Forms.PaintEventHandler(this.imageBox1_Paint);
			this.imageBox1.MouseDown += new System.Windows.Forms.MouseEventHandler(this.imageBox1_MouseDown);
			this.imageBox1.MouseMove += new System.Windows.Forms.MouseEventHandler(this.imageBox1_MouseMove);
			this.imageBox1.MouseUp += new System.Windows.Forms.MouseEventHandler(this.imageBox1_MouseUp);
			// 
			// label5
			// 
			resources.ApplyResources(this.label5, "label5");
			this.label5.Name = "label5";
			// 
			// lblBox
			// 
			this.lblBox.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
			resources.ApplyResources(this.lblBox, "lblBox");
			this.lblBox.Name = "lblBox";
			// 
			// label6
			// 
			resources.ApplyResources(this.label6, "label6");
			this.label6.Name = "label6";
			// 
			// buttonReset
			// 
			resources.ApplyResources(this.buttonReset, "buttonReset");
			this.buttonReset.Name = "buttonReset";
			this.buttonReset.UseVisualStyleBackColor = true;
			this.buttonReset.Click += new System.EventHandler(this.buttonReset_Click);
			// 
			// panel1
			// 
			this.panel1.BackColor = System.Drawing.SystemColors.ButtonFace;
			this.panel1.Controls.Add(this.button4);
			this.panel1.Controls.Add(this.button3);
			this.panel1.Controls.Add(this.button2);
			this.panel1.Controls.Add(this.button1);
			this.panel1.Controls.Add(this.lblFrame);
			this.panel1.Controls.Add(this.buttonPrev);
			this.panel1.Controls.Add(this.buttonNext);
			this.panel1.Controls.Add(this.buttonPlay);
			this.panel1.Controls.Add(this.imgDone);
			this.panel1.Controls.Add(this.hScrollBar1);
			resources.ApplyResources(this.panel1, "panel1");
			this.panel1.Name = "panel1";
			// 
			// button4
			// 
			resources.ApplyResources(this.button4, "button4");
			this.button4.Name = "button4";
			this.button4.UseVisualStyleBackColor = true;
			this.button4.Click += new System.EventHandler(this.button4_Click);
			// 
			// button3
			// 
			resources.ApplyResources(this.button3, "button3");
			this.button3.Name = "button3";
			this.button3.UseVisualStyleBackColor = true;
			this.button3.Click += new System.EventHandler(this.button3_Click);
			// 
			// button2
			// 
			resources.ApplyResources(this.button2, "button2");
			this.button2.Name = "button2";
			this.button2.UseVisualStyleBackColor = true;
			this.button2.Click += new System.EventHandler(this.button2_Click);
			// 
			// button1
			// 
			resources.ApplyResources(this.button1, "button1");
			this.button1.Name = "button1";
			this.button1.UseVisualStyleBackColor = true;
			this.button1.Click += new System.EventHandler(this.button1_Click_1);
			// 
			// lblFrame
			// 
			resources.ApplyResources(this.lblFrame, "lblFrame");
			this.lblFrame.BackColor = System.Drawing.SystemColors.ButtonFace;
			this.lblFrame.BorderStyle = System.Windows.Forms.BorderStyle.None;
			this.lblFrame.ForeColor = System.Drawing.SystemColors.ControlDarkDark;
			this.lblFrame.Name = "lblFrame";
			// 
			// buttonPrev
			// 
			resources.ApplyResources(this.buttonPrev, "buttonPrev");
			this.buttonPrev.Name = "buttonPrev";
			this.buttonPrev.UseVisualStyleBackColor = true;
			this.buttonPrev.Click += new System.EventHandler(this.buttonPrev_Click);
			// 
			// buttonNext
			// 
			resources.ApplyResources(this.buttonNext, "buttonNext");
			this.buttonNext.Name = "buttonNext";
			this.buttonNext.UseVisualStyleBackColor = true;
			this.buttonNext.Click += new System.EventHandler(this.buttonNext_Click);
			// 
			// buttonPlay
			// 
			resources.ApplyResources(this.buttonPlay, "buttonPlay");
			this.buttonPlay.Name = "buttonPlay";
			this.buttonPlay.UseVisualStyleBackColor = true;
			this.buttonPlay.Click += new System.EventHandler(this.buttonPlay_Click);
			// 
			// imgDone
			// 
			resources.ApplyResources(this.imgDone, "imgDone");
			this.imgDone.BackColor = System.Drawing.SystemColors.ButtonFace;
			this.imgDone.FunctionalMode = Emgu.CV.UI.ImageBox.FunctionalModeOption.Minimum;
			this.imgDone.Name = "imgDone";
			this.imgDone.TabStop = false;
			// 
			// hScrollBar1
			// 
			resources.ApplyResources(this.hScrollBar1, "hScrollBar1");
			this.hScrollBar1.LargeChange = 1;
			this.hScrollBar1.Maximum = 0;
			this.hScrollBar1.Name = "hScrollBar1";
			this.hScrollBar1.Scroll += new System.Windows.Forms.ScrollEventHandler(this.hScrollBar1_Scroll);
			// 
			// panel2
			// 
			this.panel2.Controls.Add(this.button5);
			this.panel2.Controls.Add(this.lblFile);
			this.panel2.Controls.Add(this.lblDir);
			this.panel2.Controls.Add(this.lblMode);
			this.panel2.Controls.Add(this.txtUser);
			this.panel2.Controls.Add(this.label7);
			this.panel2.Controls.Add(this.buttonLoad);
			this.panel2.Controls.Add(this.buttSave);
			this.panel2.Controls.Add(this.buttonReset);
			this.panel2.Controls.Add(this.label1);
			this.panel2.Controls.Add(this.label6);
			this.panel2.Controls.Add(this.label2);
			this.panel2.Controls.Add(this.lblBox);
			this.panel2.Controls.Add(this.label5);
			this.panel2.Controls.Add(this.label4);
			this.panel2.Controls.Add(this.lblMem);
			resources.ApplyResources(this.panel2, "panel2");
			this.panel2.Name = "panel2";
			// 
			// button5
			// 
			resources.ApplyResources(this.button5, "button5");
			this.button5.Name = "button5";
			this.button5.UseVisualStyleBackColor = true;
			this.button5.Click += new System.EventHandler(this.button5_Click);
			// 
			// lblFile
			// 
			this.lblFile.BackColor = System.Drawing.SystemColors.ControlLight;
			resources.ApplyResources(this.lblFile, "lblFile");
			this.lblFile.Name = "lblFile";
			this.lblFile.ReadOnly = true;
			// 
			// lblDir
			// 
			this.lblDir.BackColor = System.Drawing.SystemColors.ControlLight;
			resources.ApplyResources(this.lblDir, "lblDir");
			this.lblDir.Name = "lblDir";
			this.lblDir.ReadOnly = true;
			// 
			// lblMode
			// 
			this.lblMode.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
			resources.ApplyResources(this.lblMode, "lblMode");
			this.lblMode.Name = "lblMode";
			// 
			// txtUser
			// 
			resources.ApplyResources(this.txtUser, "txtUser");
			this.txtUser.Name = "txtUser";
			// 
			// label7
			// 
			resources.ApplyResources(this.label7, "label7");
			this.label7.Name = "label7";
			// 
			// timer1
			// 
			this.timer1.Tick += new System.EventHandler(this.timer1_Tick);
			// 
			// Form1
			// 
			resources.ApplyResources(this, "$this");
			this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
			this.BackColor = System.Drawing.SystemColors.ControlLight;
			this.Controls.Add(this.panel2);
			this.Controls.Add(this.panel1);
			this.Controls.Add(this.imageBox1);
			this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
			this.MaximizeBox = false;
			this.Name = "Form1";
			this.Load += new System.EventHandler(this.Form1_Load);
			((System.ComponentModel.ISupportInitialize)(this.imageBox1)).EndInit();
			this.panel1.ResumeLayout(false);
			this.panel1.PerformLayout();
			((System.ComponentModel.ISupportInitialize)(this.imgDone)).EndInit();
			this.panel2.ResumeLayout(false);
			this.panel2.PerformLayout();
			this.ResumeLayout(false);

		}

		#endregion
		private System.Windows.Forms.Button buttonLoad;
		private System.Windows.Forms.Button buttSave;
		private System.Windows.Forms.ImageList imageList1;
		private System.Windows.Forms.OpenFileDialog openFileDialog1;
		private System.Windows.Forms.Label label1;
		private System.Windows.Forms.Label label2;
		private System.Windows.Forms.Label label4;
		private System.Windows.Forms.Label lblMem;
		private Emgu.CV.UI.ImageBox imageBox1;
		private System.Windows.Forms.Label label5;
		private System.Windows.Forms.Label lblBox;
		private System.Windows.Forms.Label label6;
		private System.Windows.Forms.Button buttonReset;
		private System.Windows.Forms.Panel panel1;
		private System.Windows.Forms.Button buttonPrev;
		private System.Windows.Forms.Button buttonNext;
		private System.Windows.Forms.Button buttonPlay;
		private Emgu.CV.UI.ImageBox imgDone;
		private System.Windows.Forms.HScrollBar hScrollBar1;
		private System.Windows.Forms.Panel panel2;
		private System.Windows.Forms.Timer timer1;
		private System.Windows.Forms.TextBox lblFrame;
		private System.Windows.Forms.TextBox txtUser;
		private System.Windows.Forms.Label label7;
		private System.Windows.Forms.Label lblMode;
		private System.Windows.Forms.Button button2;
		private System.Windows.Forms.Button button1;
		private System.Windows.Forms.TextBox lblFile;
		private System.Windows.Forms.TextBox lblDir;
		private System.Windows.Forms.Button button4;
		private System.Windows.Forms.Button button3;
		private System.Windows.Forms.Button button5;
	}
}

