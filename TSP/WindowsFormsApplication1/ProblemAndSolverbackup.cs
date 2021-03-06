using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Drawing;
using System.Diagnostics;
using System.Linq;


namespace TSP
{

	class ProblemAndSolver
	{

		public class TSPSolution
		{
			/// <summary>
			/// we use the representation [cityB,cityA,cityC] 
			/// to mean that cityB is the first city in the solution, cityA is the second, cityC is the third 
			/// and the edge from cityC to cityB is the final edge in the path.  
			/// You are, of course, free to use a different representation if it would be more convenient or efficient 
			/// for your data structure(s) and search algorithm. 
			/// </summary>
			public ArrayList
				Route;

			/// <summary>
			/// constructor
			/// </summary>
			/// <param name="iroute">a (hopefully) valid tour</param>
			public TSPSolution(ArrayList iroute)
			{
				Route = new ArrayList(iroute);
			}

			/// <summary>
			/// Compute the cost of the current route.  
			/// Note: This does not check that the route is complete.
			/// It assumes that the route passes from the last city back to the first city. 
			/// </summary>
			/// <returns></returns>
			public double costOfRoute()
			{
				// go through each edge in the route and add up the cost. 
				int x;
				City here;
				double cost = 0D;

				for (x = 0; x < Route.Count - 1; x++)
				{
					here = Route[x] as City;
					cost += here.costToGetTo(Route[x + 1] as City);
				}

				// go from the last city to the first. 
				here = Route[Route.Count - 1] as City;
				cost += here.costToGetTo(Route[0] as City);
				return cost;
			}
		}

		#region Private members 

		/// <summary>
		/// Default number of cities (unused -- to set defaults, change the values in the GUI form)
		/// </summary>
		// (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
		// click on the Problem Size text box, go to the Properties window (lower right corner), 
		// and change the "Text" value.)
		private const int DEFAULT_SIZE = 25;

		/// <summary>
		/// Default time limit (unused -- to set defaults, change the values in the GUI form)
		/// </summary>
		// (This is no longer used -- to set default values, edit the form directly.  Open Form1.cs,
		// click on the Time text box, go to the Properties window (lower right corner), 
		// and change the "Text" value.)
		private const int TIME_LIMIT = 60;        //in seconds

		private const int CITY_ICON_SIZE = 5;


		// For normal and hard modes:
		// hard mode only
		private const double FRACTION_OF_PATHS_TO_REMOVE = 0.20;

		/// <summary>
		/// the cities in the current problem.
		/// </summary>
		private City[] Cities;
		/// <summary>
		/// a route through the current problem, useful as a temporary variable. 
		/// </summary>
		private ArrayList Route;
		/// <summary>
		/// best solution so far. 
		/// </summary>
		private TSPSolution bssf;

		/// <summary>
		/// how to color various things. 
		/// </summary>
		private Brush cityBrushStartStyle;
		private Brush cityBrushStyle;
		private Pen routePenStyle;


		/// <summary>
		/// keep track of the seed value so that the same sequence of problems can be 
		/// regenerated next time the generator is run. 
		/// </summary>
		private int _seed;
		/// <summary>
		/// number of cities to include in a problem. 
		/// </summary>
		private int _size;

		/// <summary>
		/// Difficulty level
		/// </summary>
		private HardMode.Modes _mode;

		/// <summary>
		/// random number generator. 
		/// </summary>
		private Random rnd;

		/// <summary>
		/// time limit in milliseconds for state space search
		/// can be used by any solver method to truncate the search and return the BSSF
		/// </summary>
		private int time_limit;
		#endregion

		#region Public members

		/// <summary>
		/// These three constants are used for convenience/clarity in populating and accessing the results array that is passed back to the calling Form
		/// </summary>
		public const int COST = 0;
		public const int TIME = 1;
		public const int COUNT = 2;

		public int Size
		{
			get { return _size; }
		}

		public int Seed
		{
			get { return _seed; }
		}
		#endregion

		#region Constructors
		public ProblemAndSolver()
		{
			this._seed = 1;
			rnd = new Random(1);
			this._size = DEFAULT_SIZE;
			this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

			this.resetData();
		}

		public ProblemAndSolver(int seed)
		{
			this._seed = seed;
			rnd = new Random(seed);
			this._size = DEFAULT_SIZE;
			this.time_limit = TIME_LIMIT * 1000;                  // TIME_LIMIT is in seconds, but timer wants it in milliseconds

			this.resetData();
		}

		public ProblemAndSolver(int seed, int size)
		{
			this._seed = seed;
			this._size = size;
			rnd = new Random(seed);
			this.time_limit = TIME_LIMIT * 1000;                        // TIME_LIMIT is in seconds, but timer wants it in milliseconds

			this.resetData();
		}
		public ProblemAndSolver(int seed, int size, int time)
		{
			this._seed = seed;
			this._size = size;
			rnd = new Random(seed);
			this.time_limit = time * 1000;                        // time is entered in the GUI in seconds, but timer wants it in milliseconds

			this.resetData();
		}
		#endregion

		#region Private Methods

		/// <summary>
		/// Reset the problem instance.
		/// </summary>
		private void resetData()
		{

			Cities = new City[_size];
			Route = new ArrayList(_size);
			bssf = null;

			if (_mode == HardMode.Modes.Easy)
			{
				for (int i = 0; i < _size; i++)
					Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble());
			}
			else // Medium and hard
			{
				for (int i = 0; i < _size; i++)
					Cities[i] = new City(rnd.NextDouble(), rnd.NextDouble(), rnd.NextDouble() * City.MAX_ELEVATION);
			}

			HardMode mm = new HardMode(this._mode, this.rnd, Cities);
			if (_mode == HardMode.Modes.Hard)
			{
				int edgesToRemove = (int)(_size * FRACTION_OF_PATHS_TO_REMOVE);
				mm.removePaths(edgesToRemove);
			}
			City.setModeManager(mm);

			cityBrushStyle = new SolidBrush(Color.Black);
			cityBrushStartStyle = new SolidBrush(Color.Red);
			routePenStyle = new Pen(Color.Blue, 1);
			routePenStyle.DashStyle = System.Drawing.Drawing2D.DashStyle.Solid;
		}

		#endregion

		#region Public Methods

		/// <summary>
		/// make a new problem with the given size.
		/// </summary>
		/// <param name="size">number of cities</param>
		public void GenerateProblem(int size, HardMode.Modes mode)
		{
			this._size = size;
			this._mode = mode;
			resetData();
		}

		/// <summary>
		/// make a new problem with the given size, now including timelimit paremeter that was added to form.
		/// </summary>
		/// <param name="size">number of cities</param>
		public void GenerateProblem(int size, HardMode.Modes mode, int timelimit)
		{
			this._size = size;
			this._mode = mode;
			this.time_limit = timelimit * 1000;                                   //convert seconds to milliseconds
			resetData();
		}

		/// <summary>
		/// return a copy of the cities in this problem. 
		/// </summary>
		/// <returns>array of cities</returns>
		public City[] GetCities()
		{
			City[] retCities = new City[Cities.Length];
			Array.Copy(Cities, retCities, Cities.Length);
			return retCities;
		}

		/// <summary>
		/// draw the cities in the problem.  if the bssf member is defined, then
		/// draw that too. 
		/// </summary>
		/// <param name="g">where to draw the stuff</param>
		public void Draw(Graphics g)
		{
			float width = g.VisibleClipBounds.Width - 45F;
			float height = g.VisibleClipBounds.Height - 45F;
			Font labelFont = new Font("Arial", 10);

			// Draw lines
			if (bssf != null)
			{
				// make a list of points. 
				Point[] ps = new Point[bssf.Route.Count];
				int index = 0;
				foreach (City c in bssf.Route)
				{
					if (index < bssf.Route.Count - 1)
						g.DrawString(" " + index + "(" + c.costToGetTo(bssf.Route[index + 1] as City) + ")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
					else
						g.DrawString(" " + index + "(" + c.costToGetTo(bssf.Route[0] as City) + ")", labelFont, cityBrushStartStyle, new PointF((float)c.X * width + 3F, (float)c.Y * height));
					ps[index++] = new Point((int)(c.X * width) + CITY_ICON_SIZE / 2, (int)(c.Y * height) + CITY_ICON_SIZE / 2);
				}

				if (ps.Length > 0)
				{
					g.DrawLines(routePenStyle, ps);
					g.FillEllipse(cityBrushStartStyle, (float)Cities[0].X * width - 1, (float)Cities[0].Y * height - 1, CITY_ICON_SIZE + 2, CITY_ICON_SIZE + 2);
				}

				// draw the last line. 
				g.DrawLine(routePenStyle, ps[0], ps[ps.Length - 1]);
			}

			// Draw city dots
			foreach (City c in Cities)
			{
				g.FillEllipse(cityBrushStyle, (float)c.X * width, (float)c.Y * height, CITY_ICON_SIZE, CITY_ICON_SIZE);
			}

		}

		/// <summary>
		///  return the cost of the best solution so far. 
		/// </summary>
		/// <returns></returns>
		public double costOfBssf()
		{
			if (bssf != null)
				return (bssf.costOfRoute());
			else
				return -1D;
		}

		public TSPSolution GetBSSF()
		{
			return bssf;
		}

		public void updateBSSF(ArrayList route)
		{
			ArrayList cityRoute = new ArrayList();
			for (int i = 0; i < route.Count; i++)
			{
				cityRoute.Add(Cities[(int)route[i]]);
			}
			this.bssf = new TSPSolution(cityRoute);
		}

		/// <summary>
		/// This is the entry point for the default solver
		/// which just finds a valid random tour 
		/// </summary>
		/// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
		public string[] defaultSolveProblem()
		{
			int i, swap, temp, count = 0;
			string[] results = new string[3];
			int[] perm = new int[Cities.Length];
			Route = new ArrayList();
			Stopwatch timer = new Stopwatch();

			timer.Start();

			do
			{
				for (i = 0; i < perm.Length; i++)                                 // create a random permutation template
					perm[i] = i;
				for (i = 0; i < perm.Length; i++)
				{
					swap = i;
					while (swap == i)
						swap = rnd.Next(0, Cities.Length);
					temp = perm[i];
					perm[i] = perm[swap];
					perm[swap] = temp;
				}
				Route.Clear();
				for (i = 0; i < Cities.Length; i++)                            // Now build the route using the random permutation 
				{
					Route.Add(Cities[perm[i]]);
				}
				bssf = new TSPSolution(Route);
				count++;
			} while (costOfBssf() == double.PositiveInfinity);                // until a valid route is found
			timer.Stop();

			results[COST] = costOfBssf().ToString();                          // load results array
			results[TIME] = timer.Elapsed.ToString();
			results[COUNT] = count.ToString();

			return results;
		}

		public class BBSolver
		{
			private ProblemAndSolver parent;
			private PriorityQueue pq;
			private int solCount;
			private int maxQ;
			private int totalStates;
			private int prunedStates;
			private Stopwatch stopwatch;

			public BBSolver(ProblemAndSolver parent)
			{
				this.parent = parent;
				solCount = 0;
				maxQ = 0;
				totalStates = 0;
				prunedStates = 0;
				stopwatch = new Stopwatch();
			}

			public class stateNode : IComparable<stateNode>
			{
				protected int depth; //depth in the solution tree
				protected double key; //will be calcuated from cost and depth
				private ArrayList route; //partial route of cities identified by index in Cities array list
				private ArrayList remaining;
				private CostMatrix cost_mat; //includes both matrix and total cost
				private double cost_bound; //total cost of redcued matrix

				public stateNode(int depth, ArrayList route, ArrayList remaining, CostMatrix cost_mat)
				{
					this.depth = depth;
					this.route = route;
					this.remaining = remaining;
					this.cost_mat = cost_mat;
					cost_bound = cost_mat.getCost();
					key = cost_bound / depth;
				}

				public int CompareTo(stateNode n)
				{

					if (this.key < n.key) return -1;
					else if (this.key == n.key) return 0;
					else return 1;
				}
				public double getKey()
				{
					return key;
				}
				public ArrayList getRoute()
				{
					ArrayList rou = new ArrayList();
					rou.AddRange(route);
					/* for(int i = 0; i < route.Count; i++)
                    {
                        rou.Add((int)route[i]);
                    }
                    */
					return rou;
				}
				public ArrayList getRemaining()
				{
					ArrayList rem = new ArrayList(remaining.Count);
					rem.AddRange(remaining);
					return rem;
				}
				public CostMatrix getMatrix()
				{
					return cost_mat;
				}
				public double getBound()
				{
					return cost_bound;
				}
				public int getDepth()
				{
					return depth;
				}
			}

			public class PriorityQueue
			{
				//protected List<stateNode> nodes;
				protected ArrayList nodes;
				protected ArrayList priorityQ;
				protected int pq_length;
				//protected int startnode;

				public PriorityQueue(ArrayList nodes)
				{
					this.priorityQ = (ArrayList)nodes.Clone();
					pq_length = nodes.Count;
					//makequeue(); not necessary for this problem; must reimplement if needed
				}

				public double getLength()
				{
					return pq_length;
				}

				public stateNode getNode(int idx)
				{
					return (stateNode)nodes[idx];
				}
				/*
                public double getAtIdx(int i)
                {
                    return priorityQ[i];
                }
                */
				public int getNodeCount()
				{
					return nodes.Count;
				}
				public void nullNode(int idx)
				{
					nodes[idx] = null;
				}

				/*
                public void makequeue() //overall O(n log n)
                {
                    priorityQ = new int[nodes.Count];
                    for (int i = 0; i < nodes.Count; i++) //O(n)
                    {
                        priorityQ[i] = i;
                    }

                    for (int i = nodes.Count - 1; i >= 0; i--) //O(n log n)
                    {
                        siftdown(priorityQ[i], i);
                    }

                }
                
                
                public void addNode(stateNode node)
                {
                    //priorityQ = priorityQ.Concat(new[] { nodes.Count - 1 }).ToArray();
                    pq_length++;
                    insert(node);
                }
                */
				public stateNode deletemin() //O(log n) due to the big O of siftdown
				{
					if (pq_length == 0)
					{
						return null;
					}
					else
					{
						stateNode node = (stateNode)priorityQ[0];
						pq_length--; //needs to be decremented for internal of siftdown, so that the last position isn't used now that it's being shifted
						siftdown((stateNode)priorityQ[pq_length], 0); //techincially still needed the original last position here though
						priorityQ[pq_length] = null;

						return node;
					}
				}


				public void bubbleup(stateNode node, int i)
				{
					int p = (Convert.ToInt32(Math.Ceiling(i / 2d))) - 1;
					if (p >= 0)
					{
						stateNode parent = (stateNode)priorityQ[p];
						while (p >= 0 && i > 0 && parent.getKey() > node.getKey()) //O(log n) due to halving p every time
						{
							priorityQ[i] = priorityQ[p];
							i = p;
							p = Convert.ToInt32(Math.Ceiling(i / 2d)) - 1;
							if (p >= 0) { parent = (stateNode)priorityQ[p]; }
						}
					}

					priorityQ[i] = node;
				}

				public void siftdown(stateNode node, int i)
				{
					int c = minchild(i);
					stateNode child = (stateNode)priorityQ[c];
					while (c != 0 && child.getKey() < node.getKey()) //O(log n) due to minchild moving along the height of the tree, or log n
					{
						priorityQ[i] = priorityQ[c];
						i = c;
						c = minchild(i);
						child = (stateNode)priorityQ[c];
					}
					priorityQ[i] = node;
				}

				public void insert(stateNode n) //O(log n) due to the big O of bubbleup
				{
					if (pq_length == 0)
					{
						priorityQ[pq_length] = n;
					}
					else
					{
						priorityQ.Add(n);
					}

					pq_length++;
					if (pq_length > 1)
					{
						bubbleup(n, pq_length - 1);
					}
				}

				public int minchild(int i)
				{
					if ((2 * i + 1) > pq_length - 1)
					{
						return 0; //no children
					}
					else
					{
						int left_idx = 2 * i + 1; //+1 to account for 0-based indexing
						stateNode leftchild = (stateNode)priorityQ[left_idx]; //priorityq index -> nodes index -> actual node 
						double leftchild_key = leftchild.getKey();

						int right_idx = Math.Min(pq_length - 1, (2 * i + 2)); //+2 to account for 0-based indexing
						stateNode rightchild = (stateNode)priorityQ[right_idx];
						double rightchild_key = rightchild.getKey();

						if (rightchild_key < leftchild_key)
						{
							return right_idx;
						}
						else
						{
							return left_idx;
						}
					}
				}


			}
			public class CostMatrix
			{
				private double[,] matrix;
				private double tot_cost;
				public CostMatrix(double[,] matrix, double tot_cost)
				{
					this.matrix = matrix;
					this.tot_cost = tot_cost;
				}
				public double[,] getMatrix()
				{
					double[,] mat = new double[matrix.GetLength(0), matrix.GetLength(1)];
					for (int i = 0; i < matrix.GetLength(0); i++)
					{
						for (int j = 0; j < matrix.GetLength(0); j++)
						{
							mat[i, j] = matrix[i, j];
						}
					}
					return mat;
				}
				public double getCost()
				{
					return tot_cost;
				}
				public override string ToString()
				{
					StringBuilder sb = new StringBuilder();
					for (int i = 0; i < matrix.GetLength(0); i++)
					{
						string row = "";
						for (int j = 0; j < matrix.GetLength(1); j++)
						{
							row += matrix[i, j].ToString() + " ";
						}
						sb.Append(row + "\n");
					}
					return sb.ToString();
				}
				/*
                public void generateCostMatrix()
                {
                    matrix = new double[Cities.Length, Cities.Length];

                    for (int i = 0; i < Cities.Length; i++)
                    {
                        for (int j = 0; j < Cities.Length; j++)
                        {
                            if (i == j)
                            {
                                matrix[i, j] = double.PositiveInfinity;
                            }
                            else
                            {
                                matrix[i, j] = Cities[i].costToGetTo(Cities[j]);
                            }
                        }
                    }
                }
                */


			}
			//reduceMatrix: O(4n^2) = O(n^2)
			public CostMatrix reduceMatrix(CostMatrix cost_matrix, int from, int to)
			{
				double[,] cost = cost_matrix.getMatrix();
				double totCost = cost_matrix.getCost();
				int[] zerocols = new int[cost.GetLength(1)];
				//for each row, find min; subtract min from every cell; O(n*2n) 
				for (int i = 0; i < cost.GetLength(0); i++)
				{
					double min = double.PositiveInfinity;
					//find min
					for (int j = 0; j < cost.GetLength(1); j++)
					{
						if (cost[i, j] < min)
						{
							min = cost[i, j];
						}
					}
					//if row wasn't a cancelled out row from previous reductions
					if (min != double.PositiveInfinity)
					{
						//add row's cost to total cost
						totCost += min;
						//subtract min from each cell
						for (int j = 0; j < cost.GetLength(1); j++)
						{
							if (cost[i, j] != double.PositiveInfinity)
							{
								cost[i, j] -= min;
							}
							//if there is a zero, mark the column
							if (cost[i, j] == 0)
							{
								zerocols[j] = 1;
							}
						}
					}
				}
				//for every column, if it does not have a zero, find and subtract min ; O(n*2n) 
				for (int j = 0; j < cost.GetLength(1); j++)
				{
					if (zerocols[j] != 1)
					{
						double min2 = double.PositiveInfinity;
						for (int k = 0; k < cost.GetLength(0); k++)
						{
							if (cost[k, j] < min2)
							{
								min2 = cost[k, j];
							}
						}
						//if it wasn't a cancelled out column
						if (min2 != double.PositiveInfinity)
						{
							totCost += min2;
							for (int k = 0; k < cost.GetLength(0); k++)
							{
								if (cost[k, j] != double.PositiveInfinity)
								{
									cost[k, j] -= min2;
								}
							}
						}
					}
				}
				if (from != -1)
				{
					totCost += cost[from, to];
					cost[from, to] = double.PositiveInfinity;
					//set from row to infinity
					for (int f = 0; f < cost.GetLength(1); f++)
					{
						cost[from, f] = double.PositiveInfinity;
					}
					//set to column to infinty
					for (int t = 0; t < cost.GetLength(0); t++)
					{
						cost[t, to] = double.PositiveInfinity;
					}
				}
				CostMatrix new_cost_matrix = new CostMatrix(cost, totCost);
				return new_cost_matrix;
			}

			public void makePQ(ArrayList nodes)
			{
				pq = new PriorityQueue(nodes);
			}

			public void BB(stateNode node)
			{
				double upperlimit = parent.GetBSSF().costOfRoute();
				ArrayList remaining = node.getRemaining();
				int depth = node.getDepth();
				if (remaining.Count == 0)
				{
					if (node.getBound() < upperlimit)
					{
						parent.updateBSSF(node.getRoute());
						solCount++;
					}
					//hit a leaf: check if it's better than bssf and update
					return;
				}
				//expand nodes
				for (int i = 0; i < remaining.Count; i++)
				{
					//add to route
					ArrayList oldroute = node.getRoute();
					ArrayList newroute = node.getRoute();
					newroute.Add(remaining[i]);
					int to = (int)remaining[i];
					//remove city from remaining
					ArrayList newRem = (ArrayList)remaining.Clone();
					newRem.RemoveRange(i, 1);
					//grab last city in parent node's route
					int from = (int)oldroute[oldroute.Count - 1];

					//reduce cost matrix and create child node
					CostMatrix costmat = reduceMatrix(node.getMatrix(), from, to);
					stateNode newNode = new stateNode(depth + 1, newroute, newRem, costmat);
					if (newNode.getBound() < upperlimit)
					{
						pq.insert(newNode);
					}
				}
				//pq.nullNode(pos);
				//deletemin, check that it's better than bssf and run BB
				if (pq.getLength() != 0 && stopwatch.Elapsed.CompareTo(new TimeSpan(0, 0, 60)) <= 0)
				//if (pq.getLength() != 0)
				{
					stateNode nextNode = pq.deletemin();
					while (nextNode != null && nextNode.getBound() >= upperlimit)
					{
						nextNode = pq.deletemin();
					}
					if (nextNode != null)
					{
						BB(nextNode);
					}
					else return;

				}
				else return;


				//also have to keep track of # of solutionns etc



			}
			/// <summary>
			/// performs a Branch and Bound search of the state space of partial tours
			/// stops when time limit expires and uses BSSF as solution
			/// </summary>
			/// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>

			public string[] runBB()
			{
				string[] results = new string[3];

				//actual method
				//calculate cost matrix; O(n^2)
				double[,] cost = new double[parent.Cities.Length, parent.Cities.Length];

				for (int i = 0; i < parent.Cities.Length; i++)
				{
					for (int j = 0; j < parent.Cities.Length; j++)
					{
						if (i == j)
						{
							cost[i, j] = double.PositiveInfinity;
						}
						else
						{
							cost[i, j] = parent.Cities[i].costToGetTo(parent.Cities[j]);
						}
					}
				}

				CostMatrix init_matrix = new CostMatrix(cost, 0);

				parent.defaultSolveProblem(); //set bssf

				ArrayList remainingCities = new ArrayList(parent.Cities.Length);
				for (int i = 1; i < parent.Cities.Length; i++)
				{
					remainingCities.Add(i);
				}

				ArrayList start_city = new ArrayList();
				start_city.Add(0);
				CostMatrix start_matrix = reduceMatrix(init_matrix, -1, -1);

				// ArrayList nodes = new ArrayList();
				//nodes.Add(new stateNode(1, start_city, remainingCities, start_matrix));

				//pq = new PriorityQueue(nodes);
				var pq = new C5.IntervalHeap<stateNode>();
				pq.Add(new stateNode(1, start_city, remainingCities, start_matrix)); //O(log n)
				totalStates++;

				stopwatch.Start();
				//while there are statenodes on the priority queue, deletemin and expand the node 
				//for each unvisited city; update bssf when a leaf is reached; check each node against bssf before expanding.
				//worst case O(n^2*2^n)
				while (pq.Count() > 0 && stopwatch.Elapsed.CompareTo(new TimeSpan(0, 0, 60)) <= 0)
				{
					stateNode node = pq.DeleteMin(); //O(log n)
					double upperlimit = parent.GetBSSF().costOfRoute();
					if (node.getBound() < upperlimit)
					{
						//BB(node);

						//upperlimit = parent.GetBSSF().costOfRoute();
						ArrayList remaining = node.getRemaining();
						int depth = node.getRoute().Count;
						if (remaining.Count == 0)
						{
							if (node.getBound() < upperlimit)
							{
								parent.updateBSSF(node.getRoute());
								solCount++;
							}
							//hit a leaf: check if it's better than bssf and update
							// break;
						}
						else
						{
							//expand nodes
							for (int i = 0; i < remaining.Count; i++)
							{
								//add to route
								ArrayList oldroute = node.getRoute();
								ArrayList newroute = node.getRoute();
								newroute.Add(remaining[i]);
								int to = (int)remaining[i];
								//remove city from remaining
								ArrayList newRem = (ArrayList)remaining.Clone();
								newRem.RemoveRange(i, 1);
								//grab last city in parent node's route
								int from = (int)oldroute[oldroute.Count - 1];

								//reduce cost matrix and create child node
								CostMatrix costmat = reduceMatrix(node.getMatrix(), from, to); //O(n^2)
								stateNode newNode = new stateNode(depth + 1, newroute, newRem, costmat);
								totalStates++;
								if (newNode.getBound() < upperlimit)
								{
									pq.Add(newNode);
									if (pq.Count > maxQ)
									{
										maxQ = pq.Count;
									}
									else prunedStates++;
								}
							}
						}
					}
					else prunedStates++;
				}
				stopwatch.Stop();
				TimeSpan el_time = stopwatch.Elapsed;

				//prunedStates += pq.Count;

				results[COST] = parent.costOfBssf().ToString();
				results[TIME] = el_time.ToString();
				results[COUNT] = solCount.ToString();
				Console.WriteLine("max Q length: " + maxQ.ToString());
				Console.WriteLine("total states created: " + totalStates.ToString());
				Console.WriteLine("states pruned: " + prunedStates.ToString());


				return results;
			}
		}

		public string[] bBSolveProblem()
		{
			BBSolver bb = new BBSolver(this);
			//string[] results = new string[3];

			string[] results = bb.runBB();

			/*
            results[COST] = "not implemented";    // load results into array here, replacing these dummy values
            results[TIME] = "-1";
            results[COUNT] = "-1";
            */

			return results;
		}

		/////////////////////////////////////////////////////////////////////////////////////////////
		// These additional solver methods will be implemented as part of the group project.
		////////////////////////////////////////////////////////////////////////////////////////////

		/// <summary>
		/// finds the greedy tour starting from each city and keeps the best (valid) one
		/// </summary>
		/// <returns>results array for GUI that contains three ints: cost of solution, time spent to find solution, number of solutions found during search (not counting initial BSSF estimate)</returns>
		public string[] greedySolveProblem()
		{
			string[] results = new string[3];

			// TODO: Add your implementation for a greedy solver here.
			Stopwatch timer = new Stopwatch();
			timer.Start();
			var list = greedyUpperBound(new List<City>(Cities), Cities[0]);
			bssf = new TSPSolution(new ArrayList(list));
			timer.Stop();
			results[COST] = bssf.costOfRoute().ToString();    // load results into array here, replacing these dummy values
			results[TIME] = timer.ToString();
			results[COUNT] = "1";

			return results;
		}

		private ArrayList greedyUpperBound(List<City> remaining, City current)
		{
			if (remaining.Count == 0)
			{
				var list = new ArrayList(Cities.Length);
				list.Add(current);
				return list;
			}

			var priorities = remaining.OrderBy(x => current.costToGetTo(x));
			foreach (var city in priorities)
			{
				if (current.costToGetTo(city) < double.PositiveInfinity)
				{
					remaining.Remove(city);
					ArrayList list = greedyUpperBound(remaining, city);
					if (list != null)
					{
						list.Add(current);
						return list;
					}
					remaining.Insert(0, city);
				}
			}


			return null;
		}

		public class Chromosome
		{
			private ArrayList route;
			public double fitness;
			public Chromosome(TSPSolution sol)
			{
				this.route = sol.Route;
				this.fitness = sol.costOfRoute();
			}
			public double getFitness()
			{
				return fitness;
			}
			public ArrayList getRoute()
			{
				return route;
			}
			public void reCalcFitness()
			{

			}
		}

		public List<Chromosome> weighted_dice(List<Chromosome> solutions, int returnNumber)
		{
			//function will take in n number of solutions 
			//and pick 10 and send them back
			//the rest will be pruned

			//order the list of solutions
			List<Chromosome> winners = new List<Chromosome>();
			//order the solutions in ascending order
			List<Chromosome> SortedList = solutions.OrderBy(o => o.fitness).ToList();
			Random random = new Random();

			int size = solutions.Count;
			if (size <= returnNumber)
			{
				return SortedList;
			}
			else if (size < 20) //if number of solutions is small, just pass back the best ten
			{
				for (int i = 0; i < 10; i++)
				{
					winners.Add(SortedList[i]);
				}
				return winners;
			}
			//other wise find 10 programatically 
			int ranNum1 = 0;

			int num1 = size / 2 / 2;
			int num2 = size / 2;
			int num3 = num2 + num1;
			int num4 = size;



			//Runs faster when list passed in is large
			while (winners.Count < returnNumber) //taking the best 10
			{
				ranNum1 = random.Next(0, 1000);

				if (ranNum1 <= 50)      //5% chance going into the bottom 1/4
				{
					//bottom 4/4
					int rand2 = random.Next(num3, num4);
					if (SortedList[rand2] != null)
					{
						winners.Add(SortedList[rand2]);
						SortedList[rand2] = null;
					}
				}
				else if (ranNum1 <= 100)        //10% chance of going into the bottom 2/4th quatile
				{
					//bottom 3/4
					int rand3 = random.Next(num2, num3);
					if (SortedList[rand3] != null)
					{
						winners.Add(SortedList[rand3]);
						SortedList[rand3] = null;
					}
				}
				else if (ranNum1 <= 150)        //15% chance going into the 2nd best quartile
				{
					//bottom 2/4
					int rand4 = random.Next(num1, num2);
					if (SortedList[rand4] != null)
					{
						winners.Add(SortedList[rand4]);
						SortedList[rand4] = null;
					}
				}
				else if (ranNum1 < 1000)        //70% chance going into the best quartile
				{
					//most likely 1/4
					int rand5 = random.Next(0, num1);
					if (SortedList[rand5] != null)
					{
						winners.Add(SortedList[rand5]);
						SortedList[rand5] = null;
					}
				}

			}
			return winners;
		}

		public class intpair
		{
			public int x;
			public int y;
			public intpair(int x, int y)
			{
				this.x = x;
				this.y = y;
			}

		}

		public List<intpair> pair_mates(int size, int numchildren)
		{
			//brain dead check for threshold for maintaining unique pairs
			if (numchildren > size)
			{
				return null;
			}
			Random rand = new Random();
			List<intpair> pairs = new List<intpair>();
			for (int i = 0; i < size; i++)
			{
				for (int j = 0; j < numchildren; j++)
				{
					int mate = rand.Next(size - 1);
					while (mate == i)
					{
						//Console.WriteLine("invalid");
						mate = rand.Next(size - 1);
					}
					pairs.Add(new intpair(i, mate));
				}
			}

			//for testing
			/*
            Console.WriteLine("pairs");
            foreach (intpair p in pairs)
            {
                int x = p.x;
                int y = p.y;
                Console.WriteLine(x.ToString() + " " + y.ToString());
            }
            */
			return pairs;

		}

		public List<Chromosome> crossover_all(List<Chromosome> pop, List<intpair> pairs)
		{

			foreach (var pair in pairs)
			{
				var child = mate(pop[pair.x], pop[pair.y]);
				if (child != null)
				{
					pop.Add(child);
				}
			}
			return pop;
		}

		public Chromosome mate(Chromosome father, Chromosome mother)
		{
			List<City> child = new List<City>(father.getRoute().Count);
			var random = new Random();
			int attempts = 0;
			do
			{
				child.Clear();
				var splitPoint = random.Next() % (father.getRoute().Count);

				for (int i = 0; i < splitPoint; i++)
				{
					child.Add((City)father.getRoute()[i]);
				}
				for (int i = splitPoint; i < mother.getRoute().Count; i++)
				{
					child.Add((City)mother.getRoute()[i]);
				}

				child = replaceDuplicates(child);

				attempts++;

				if (attempts > 100)
				{
					return null;
				}

			} while (!isValid(child));

			return new Chromosome(new TSPSolution(new ArrayList(child)));
		}

		public List<City> replaceDuplicates(List<City> child)
		{

			var groups = child.Select((x, i) => new { index = i, city = x }).GroupBy(x => x.city);
			var duplicates = groups.Where(x => x.Count() > 1).Select(group => group.First().index).ToList();
			var used = groups.Select(x => x.Key);
			var unused = Cities.Except(used).OrderBy(x => rnd.Next()).ToList();

			foreach (var i in duplicates)
			{
				for (var j = 0; j < unused.Count(); j++)
				{
					bool isValid = true;
					if (i > 0)
					{
						isValid = isValid && !double.IsPositiveInfinity(child[i - 1].costToGetTo(unused[j]));
					}
					else
					{
						isValid = isValid && !double.IsPositiveInfinity(child[child.Count() - 1].costToGetTo(unused[j]));
					}
					if (i < child.Count())
					{
						isValid = isValid && !double.IsPositiveInfinity(unused[j].costToGetTo(child[i + 1]));
					}
					else
					{
						isValid = isValid && !double.IsPositiveInfinity(unused[j].costToGetTo(child[0]));
					}
					if (isValid)
					{
						child.RemoveAt(i);
						child.Insert(i, unused[j]);
						break;
					}
				}
			}

			return child;
		}

		public bool isValid(List<City> c)
		{
			if (c.GroupBy(x => x).Any(x => x.Count() > 1))
			{
				return false;
			}

			for (var i = 0; i < c.Count - 1; i++)
			{

				if (Double.IsPositiveInfinity(c[i].costToGetTo(c[i + 1])))
				{
					return false;
				}
			}
			return !Double.IsPositiveInfinity(c[c.Count - 1].costToGetTo(c[0]));
		}

		public List<Chromosome> mutate_all(List<Chromosome> pop)
		{
			//return new generation
			int count = pop.Count;//Size wouldn't work since it grows.
			for (int i = 0; i < count; ++i)
			{
				pop.Add(mutateChromosome(pop[i]));
			}
			return pop;
		}
		private bool invalidNum(double val)
		{
			return (double.IsPositiveInfinity(val) || double.IsNaN(val));
		}
		private Chromosome mutateChromosome(Chromosome chromosome)
		{
			Chromosome result = new Chromosome(new TSPSolution(chromosome.getRoute()));
			Random rand = new Random();
			int max = chromosome.getRoute().Count / 10;
			if (max < 1) max = 1;
			int numberOfMutations = rand.Next(1, max);//Dont want too many mutations.
													  //numberOfMutations=(int)Math.Sqrt(numberOfMutations);
			if (numberOfMutations <= 0) numberOfMutations = 1;
			int x, y;
			City cx, cy, swap, cxP, cxA, cyP, cyA;
			ArrayList route = result.getRoute();
			for (int i = 0; i < numberOfMutations; ++i)
			{
				do
				{
					x = rand.Next(1, route.Count);
					y = rand.Next(1, route.Count);
					cx = (City)route[x];
					cy = (City)route[y];
					cxP = (x != 0) ? (City)(route[x - 1]) : (City)(route[route.Count - 1]);
					cxA = (x != route.Count - 1) ? (City)(route[x + 1]) : (City)(route[0]);
					cyP = (y != 0) ? (City)(route[y - 1]) : (City)(route[route.Count - 1]);
					cyA = (y != route.Count - 1) ? (City)(route[y + 1]) : (City)(route[0]);
				} while (
				invalidNum(cxP.costToGetTo(cy)) || // x.previous to y
				invalidNum(cx.costToGetTo(cyA)) || // x to y.next
				invalidNum(cyP.costToGetTo(cx)) || // y.previous to x
				invalidNum(cy.costToGetTo(cxA)) // y to x.next
				);
				swap = (City)route[x];
				route[x] = route[y];
				route[y] = swap;
			}
			return result;
		}

		public string[] fancySolveProblem()
		{
			Stopwatch timer = new Stopwatch();
			TimeSpan TIMELIMIT = new TimeSpan(0, 0, time_limit / 1000);
			int NUMCHILDREN = 3;

			int solution_count = 0;

			string[] results = new string[3];

			//Generating initial solutions
			int INITNUM = 500;
			List<Chromosome> currentpop = new List<Chromosome>();
			for (int i = 0; i < INITNUM; i++)
			{
				String[] solution = defaultSolveProblem();
				currentpop.Add(new Chromosome(bssf));
			}

			//loop
			timer.Start();
			while (timer.Elapsed.CompareTo(TIMELIMIT) <= 0)
			{
				currentpop = weighted_dice(currentpop, INITNUM);
				List<intpair> pairlist = pair_mates(currentpop.Count, NUMCHILDREN);
				currentpop = crossover_all(currentpop, pairlist);
				currentpop = mutate_all(currentpop);
				//need to figure out how to fix the fact that this will save routes that have a chance of getting deleted
				foreach (Chromosome sol in currentpop)
				{
					if (sol.getFitness() < bssf.costOfRoute())
					{
						bssf = new TSPSolution(sol.getRoute()); //assuming route is ArrayList of Cities
						solution_count++;
					}
				}

			}
			timer.Stop();
			TimeSpan el_time = timer.Elapsed;

			results[COST] = bssf.costOfRoute().ToString();
			results[TIME] = el_time.ToString();
			results[COUNT] = solution_count.ToString();

			return results;
		}
		#endregion
	}

}
