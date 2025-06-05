def add_book():
    title=input("Enter book title: ")
    author=input("Enter book author: ")
    year=input("Enter book year: ")
    with open("library.txt","a")as file:
        file.write(f"Title:{title}\nAuthor:{author}\nYear:{year}\n\n")
        print("Book added successfully.")
add_book()
print()
def view_all_book():
    try:
        with open("library.txt","r")as file:
            books=file.read()
            with open("library.txt","w")as view_file:
             view_file.write(books)
             print("View all books saved in view_all_books.txt file.")
    except FileNotFoundError:
        print("No book in the library.")
view_all_book()
print()
def Search_book():
    title=input("Enter a Book title to search: ")
    try:
         with open("library.txt","r") as file:
             for line in file:
                   if title.lower()in line.lower():
                       print(line)
                       break
             else:
                 print("Book not found")          
    except FileNotFoundError:
        print("No Book in the library.")
Search_book()
print()

def Delete_book():
    title=input("Enter  Book title to Delete: ")
    try:
        with open("library.txt","r") as file:
            line=[line for line in file if title.lower()not in line.lower()]
            with open("library.txt","w")as view_file:
                print("Book Deleted Succesfully.")       
       
    except FileNotFoundError:
        print("No Book in the library")
Delete_book()
print()

def main():
    while True:
        print("\n Library managment system")
        print("1.Add a Book")
        print("2. View All Book")
        print("3. Search a Book")
        print("4. Delete a Book")
        print("5. Exit")
        choice=input("Enter Your Choice:")
        if choice=="1":
            add_book()
        elif choice=="2":
            view_all_book()
        elif choice=="3":
            Search_book()
        elif choice=="4":
            Delete_book()
        elif choice=="5":
            print("Thank you for using library managment System!")
            print("Exiting...!")
            break
        else:
            print("Invalid Choice. Please try again.")
if __name__=="__main__":
    main()
            
         
               
            
        

            
          
      
            
           
                
            
             
    
  
    
       

